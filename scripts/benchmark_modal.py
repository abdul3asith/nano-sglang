from deployment.modal_app import generate_remote


def main() -> None:
    prompts = [
        "The capital of France is",
        "A GPU is useful for",
        "Paged attention helps because",
    ]

    for prompt in prompts:
        result = generate_remote.remote(prompt, max_new_tokens=16)
        print(f"prompt: {prompt}")
        print(f"output: {result['text']}")
        print()


if __name__ == "__main__":
    main()
