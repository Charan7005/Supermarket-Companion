import google.generativeai as genai

def summarize_with_gemini(text, user_input, api_key="AIzaSyAfgzVM9-1oblbTAPSslB_GmJIu67hmRug"):
    """Summarizes text using the Gemini API, incorporating provided connection code.

    Args:
        text: The text to summarize.
        user_input: The user-provided input for generating content.
        api_key: The Gemini API key (optional, defaults to provided one).

    Returns:
        The generated summary.
    """

    try:
        genai.configure(api_key=api_key)  
        model = genai.GenerativeModel("gemini-pro")  

        full_prompt = f"Summarize the following text:\n{text}\nAdditional input: {user_input}"

        response = model.generate_content(full_prompt) 
        return response.text

    except Exception as e:
        print(f"Error connecting to Gemini API: {e}")
        return None

text_to_summarize = """
A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can perform generic sets of operations known as programs. These programs enable computers to perform a wide range of tasks. The term computer system may refer to a nominally complete computer that includes the hardware, operating system, software, and peripheral equipment needed and used for full operation; or to a group of computers that are linked and function together, such as a computer network or computer cluster.
A broad range of industrial and consumer products use computers as control systems, including simple special-purpose devices like microwave ovens and remote controls, and factory devices like industrial robots. Computers are at the core of general-purpose devices such as personal computers and mobile devices such as smartphones. Computers power the Internet, which links billions of computers and users."""
user_input = "Please keep the summary concise."

summary = summarize_with_gemini(text_to_summarize, user_input)
print(summary)
