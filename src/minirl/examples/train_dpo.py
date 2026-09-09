"""DPO training example."""

import copy

import torch
from tqdm import tqdm
import transformers
import typer

from minirl import dpo


def _create_synthetic_preference_dataset():
    prompts = [
        "Customer service was",
        "The project deadline is approaching and",
        "When explaining complex topics to beginners,",
        "The newly released software update is",
    ] * 25  # 100 examples

    # preference: positive sentiments
    chosen = [
        " prompt, exceptionally polite, and resolved my problem completely.",
        " we are well organized and confident in delivering high quality work on time.",
        " it's best to use clear analogies, empathetic language, and structured steps.",
        " quite stable, fast, and introduces several helpful productivity features.",
    ] * 25

    # disprefer: negative sentiments
    rejected = [
        " awful, slow, and completely ignored what I asked.",
        " a disaster and everyone is stressed and blaming each other.",
        " just tell them to read the documentation and stop asking dumb questions.",
        " full of bugs, crashed my machine, and broke everything.",
    ] * 25

    return [
        dpo.PreferenceInstance(prompt=p, chosen=c, rejected=r)
        for p, c, r in zip(prompts, chosen, rejected, strict=True)
    ]


def _get_device() -> str:
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        assert device is not None
        return str(device)

    return "cpu"


def _generate_completion(
    model, tokenizer, prompt: str, max_new_tokens: int = 20
) -> str:
    model.eval()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def _compare_completions(policy_model, ref_model, tokenizer) -> None:
    prompts = {
        "In-distribution": [
            "Customer service was",
            "The project deadline is approaching and",
        ],
        "Near-domain (adjacent tech/work)": [
            "During code review, the senior engineer noted that",
            "When the client requested sudden changes to the scope,",
        ],
        "Adversity": [
            "When the server experienced unexpected downtime,",
            "When a teammate makes an honest mistake,",
        ],
        "Far-domain (everyday situations)": [
            "The hotel check-in experience was",
            "When teaching a child how to ride a bicycle,",
        ],
    }
    typer.echo("Qualitative model comparison")
    for category, prompt_list in prompts.items():
        typer.echo(f"\n--- {category} ---")
        for prompt in prompt_list:
            before = _generate_completion(ref_model, tokenizer, prompt)
            after = _generate_completion(policy_model, tokenizer, prompt)
            typer.echo(f"\nPrompt: '{prompt}'")
            typer.echo(f"  [Ref / Before]: {before.strip()}")
            typer.echo(f"  [Policy/After]: {after.strip()}")


def train(
    model_name: str = typer.Option(
        "distilbert/distilgpt2", help="HuggingFace model name or path."
    ),
    learning_rate: float = typer.Option(5e-5, help="Learning rate for AdamW."),
    beta: float = typer.Option(0.1, min=0.0, help="DPO beta parameter."),
    batch_size: int = typer.Option(
        16, min=1, help="Number of preference pairs per training batch."
    ),
    num_steps: int = typer.Option(50, min=1, help="Number of training steps."),
    device: str = typer.Option(
        "auto",
        help=(
            "Device to train on ('cpu', 'mps', 'cuda', or 'auto')."
            " 'auto' picks MPS if available, otherwise CPU."
        ),
    ),
) -> None:
    """Runs DPO alignment training on a synthetic preference dataset."""
    if device == "auto":
        device = _get_device()
    typer.echo(f"Running DPO training on {device}...")

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    assert callable(tokenizer)
    tokenizer.pad_token = tokenizer.eos_token

    policy_model = transformers.AutoModelForCausalLM.from_pretrained(model_name).to(
        device
    )
    ref_model = copy.deepcopy(policy_model).to(device)

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=learning_rate)
    trainer = dpo.DPOTrainer(policy_model, ref_model, optimizer, beta=beta)

    data = _create_synthetic_preference_dataset()
    batch = dpo.collate_batch(data[:batch_size], tokenizer)
    batch = {k: v.to(device) for k, v in batch.items()}

    typer.echo("Starting DPO Training Steps:")
    metrics = None
    pbar = tqdm(range(1, num_steps + 1), desc="Training")
    for step in pbar:
        metrics = trainer.train_step(batch)
        pbar.set_postfix(
            loss=f"{metrics.loss:.4f}", acc=f"{metrics.reward_accuracy * 100:.1f}%"
        )
        if step % 10 == 0 or step == 1:
            pbar.write(
                typer.style(
                    f"Step {step:02d}"
                    f" | Loss: {metrics.loss:.4f}"
                    f" | Margin: {metrics.reward_margin:+.4f}"
                    f" | Acc: {metrics.reward_accuracy * 100:.1f}%",
                    fg=typer.colors.CYAN,
                ),
            )

    # final step assertions
    assert metrics is not None
    assert metrics.reward_margin > 1.5, (
        f"Reward margin failed to expand: {metrics.reward_margin}"
    )
    assert metrics.reward_accuracy >= 0.90, (
        f"Accuracy under target: {metrics.reward_accuracy}"
    )
    typer.echo("DPO alignment training done.")

    # check completions between ref and policy models
    _compare_completions(policy_model, ref_model, tokenizer)


if __name__ == "__main__":
    typer.run(train)
