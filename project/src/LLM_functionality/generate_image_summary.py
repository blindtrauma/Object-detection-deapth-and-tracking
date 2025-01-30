import openai
import os
import base64


# add open ai key



# add client

def analyze_image(query, frame_link):

    textt = f'''You are an Image analyst who has to detect the IMAGE and report if there is a tool like double end combination wrench, pliers, screwdriver visisble on the screen
                If you detect thAT then respond. in certain way
                Sample Responses:
                1. The image on the screen displays a man holding a Wrench of xyz color.
                2. The image on the screen shows a plier lying on a table of xyz color.
                3. Cannot detect any mechanical tool in the image 
              "ANSWER ONLY IF YOU DETECT ANY TOOL AMONG THESE THREE :- PLIERS, DOUBLE END COMBINATION WRENCH, SCREWDRIVER, OTHERWISE RESPOND WITH THE 3RD RESPONSE IN SAMPLE RESPONSES"
              '''
              
    # Create and send the request to OpenAI API
    print(textt)
    def encode_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
    base64_image = encode_image(frame_link)
    response = client.chat.completions.create(
      model="gpt-4o",
      messages=[
        {
          "role": "user",
          "content": [
            {"type": "text", "text": textt},
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              },
            },
          ],
        }
      ],
      max_tokens=60,
    )
    print(response.choices[0].message.content, "this is the content response")
    return response.choices[0].message.content

def ai_image_analysis_and_summary(frame, query):
    print("this is being executed, ai image analysis")

    if not query:
        query = None
    target_text =analyze_image(query, frame_link=r'output_images\image.jpg')
    return target_text
