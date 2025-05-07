import nemo.collections.asr as nemo_asr
import gradio as gr

asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
    model_name="nvidia/parakeet-rnnt-1.1b"
)


def transcribe_audio(audio_file: str) -> str:
    output = asr_model.transcribe([audio_file])
    print(output[0].text)
    return output[0].text


with gr.Blocks(title="Audio Transcription Tool", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Audio Transcription Tool")
    gr.Markdown(
        "Upload an audio file (.wav format recommended) and get its transcription."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                label="Upload Audio File",
                type="filepath",
                sources=["upload", "microphone"],
            )
            with gr.Row():
                transcribe_button = gr.Button("Transcribe Audio", variant="primary")
                clear_button = gr.ClearButton(components=[audio_input])

        with gr.Column():
            output_text = gr.Textbox(
                label="Transcription Result",
                placeholder="Transcription will appear here...",
                lines=10,
            )

    transcribe_button.click(
        fn=transcribe_audio, inputs=audio_input, outputs=output_text
    )

    def clear():
        return ""

    clear_button.click(fn=clear, inputs=[], outputs=[output_text])

    gr.Markdown("""
    ## 📝 Notes
    - Best results are achieved with clear audio and minimal background noise
    - Supported formats depend on your system's audio libraries, but WAV is most reliable
    - Transcription accuracy depends on audio quality and speech clarity
    - For large files, processing may take some time
    """)

if __name__ == "__main__":
    demo.launch()
