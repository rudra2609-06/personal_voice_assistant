import base64
from google import genai
from config import GEMINI_API_KEY

client = genai.Client(api_key=GEMINI_API_KEY)


def synthesize_speech(text: str, out_path: str = "out.wav", style: str = "cheerful and friendly") -> str:
    """
	Accepts the transcipt, output_path and style
    return the file_path where audio file exists
    """
    interaction = client.interactions.create(
        model="gemini-3.8-flash-tts",
        input=[{
            "type": "user_input",
            "content": [{
                "type": "text",
                "text": text, 
                "annotations": [{
                    "type": "speech_metadata",
                    "style": style,
                }],
            }],
        }],
        response_format={"type": "audio"},
        generation_config={"speech_config": [{"voice": "Kore"}]},
    )
    with open(out_path, "wb") as f:
        f.write(base64.b64decode(interaction.output_audio.data))
    return out_path
