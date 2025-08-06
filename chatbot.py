from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def main():
    model_name = 'bigscience/gpt-oss-20b'
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map='auto',
        torch_dtype=torch.float16
    )

    print("Chatbot iniciado! Digite sua mensagem (ou 'exit' para sair)")
    chat_history = ""

    while True:
        user_input = input("Você: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Encerrando chatbot. Até mais!")
            break

        # Prepare input
        inputs = tokenizer(
            chat_history + user_input,
            return_tensors='pt'
        )
        input_ids = inputs.input_ids.to(model.device)

        # Generate response
        output_ids = model.generate(
            input_ids,
            max_length=input_ids.shape[-1] + 100,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.9,
            temperature=0.8
        )
        response = tokenizer.decode(
            output_ids[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )

        print("Bot:", response)

        # Update chat history
        chat_history += user_input + response

if __name__ == '__main__':
    main()
