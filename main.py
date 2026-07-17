import argparse

def main():
    parser = argparse.ArgumentParser(description="Sentinel UI Runner")
    parser.add_argument("--share", action="store_true", help="Gradio public link paylaşımını aç (Colab/Kaggle için)")
    parser.add_argument("--server-name", default="127.0.0.1", help="Gradio sunucu host adı")
    parser.add_argument("--server-port", type=int, default=7860, help="Gradio başlangıç portu")
    parser.add_argument("--vllm", action="store_true", help="vLLM motorunu kullan (hızlı çıkarım için, Linux/GPU gerekir)")
    args = parser.parse_args()

    from src.ui.app import build_app
    app = build_app()
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        debug=True
    )

if __name__ == "__main__":
    main()
