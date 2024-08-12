import streamlit as st
import json
from PIL import Image
from datetime import datetime
import openai
import base64
from langchain_fireworks import ChatFireworks
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_milvus import Milvus
from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

# Setting the page config to include page title and icon
st.set_page_config(page_title="Meal Maven", page_icon="mealmaven_icon.ico", layout="centered")

# Load the icon image
icon = Image.open("mealmaven_display.ico")

# Custom CSS for green color theme
st.markdown("""
    <style>
    .stSlider > div > div > div > div {
        background-color: #39A76A;  /* Green color for slider */
    }
    div.stButton > button {
        background-color: #39A76A;  /* Green color for submit button */
        color: white;
        border: none;
        border-radius: 8px;
    }
    div.stButton > button:hover {
        background-color: #2f8a5d;  /* Darker green on hover */
    }
    </style>
""", unsafe_allow_html=True)

# Define a function to save data to a JSON file
def save_to_json(data, filename="user_data.json"):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

# Define a function to read user data from the JSON file and generate a greeting message with a diet and workout plan
def generate_initial_bot_message(filename="user_data.json"):
    try:
        with open(filename, 'r') as f:
            user_data = json.load(f)

        # Generate greeting message
        greeting = f"Hello, {user_data['username']}! Welcome to Meal Maven! Based on the details you've provided, here's a diet and workout plan tailored for your goals."

        # Hardcoded diet and workout plan for now
        diet_plan = """
        \nFor your body weight and fitness goal, you can have approximately **2000 calories per day**, distributed as follows:
        \n- **Carbohydrates:** 250g
        \n- **Proteins:** 150g
        \n- **Fats:** 70g
        \n**Diet Plan:**
        \n- Breakfast: Oatmeal with fresh fruits and a glass of almond milk.
        \n- Mid-morning Snack: A handful of nuts or a protein bar.
        \n- Lunch: Grilled chicken with quinoa and steamed vegetables.
        \n- Afternoon Snack: Greek yogurt with honey and berries.
        \n- Dinner: Baked salmon with brown rice and sautéed spinach.
        \n- Evening Snack: A small apple or a serving of mixed berries.
        """

        workout_plan = """\n**Workout Plan:**
        \n- Monday: 30 minutes of cardio followed by full-body strength training.
        \n- Tuesday: 45 minutes of HIIT.
        \n- Wednesday: Rest day or light yoga/stretching.
        \n- Thursday: 30 minutes of cardio followed by lower body strength training.
        \n- Friday: 45 minutes of cycling or swimming.
        \n- Saturday: Upper body strength training.
        \n- Sunday: Rest day or a light walk.
        """

        # Combine the greeting, diet plan, and workout plan
        initial_message = f"{greeting}\n\n{diet_plan}\n\n{workout_plan}"
        return initial_message

    except FileNotFoundError:
        return "Welcome to Meal Maven! It seems we couldn't find your data. Please start by signing up!"

# Dummy function to handle updates from the user
# def call_food_option_update():
#     return "Sample_update_flow"

class Food(BaseModel):
    items: list[str] = Field(description="List of food items")


class FinalResponse(BaseModel):
    food_options: list[str] = Field(description="List of similar food options")
    food: str = Field(description="Name of the food")
    calories: int = Field(description="Number of calories")


# Helper function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def vlm_output(image_path="bagel.jpg"):
    # The base64 string of the image
    image_base64 = encode_image(image_path)

    # client = openai.OpenAI(
    #   api_key = "sk-nv-visualinsightagent-id-tuk8RSVmFbhC3Neo4EihT3BlbkFJzXcJWBrvUyYkNIacTwFt",
    # )
    model = ChatOpenAI(model="gpt-4o-mini",
                       api_key="sk-nv-visualinsightagent-id-tuk8RSVmFbhC3Neo4EihT3BlbkFJzXcJWBrvUyYkNIacTwFt")
    # model = ChatFireworks(model="accounts/fireworks/models/phi-3-vision-128k-instruct")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "what is the food in this image?"},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"},
            },
        ],
    )

    structured_llm = model.with_structured_output(Food)

    response1 = structured_llm.invoke([message])
    print(response1)

    from langchain_openai import OpenAIEmbeddings
    from langchain import hub

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                                  api_key="sk-nv-visualinsightagent-id-tuk8RSVmFbhC3Neo4EihT3BlbkFJzXcJWBrvUyYkNIacTwFt")
    connection = {"host": "127.0.0.1", "port": 19530}
    vector_db = Milvus(embedding_function=embeddings, connection_args=connection,
                       collection_name="calories",
                       auto_id=True,
                       drop_old=True)

    doc1 = Document(page_content="Number of calories in plain bagel is 180 calories")
    doc2 = Document(page_content="Number of calories in brown bread bagel is 150 calories")
    doc3 = Document(page_content="Number of calories in jalapeno cheese bagel is 230 calories")
    doc4 = Document(page_content="Number of calories in everything bagel is 200 calories")
    doc5 = Document(page_content="Number of calories in 1 tbsp of cream cheese is 70 calories")
    doc6 = Document(page_content="Number of calories in 1 tbsp of tomato basil cheese is 100 calories")
    doc7 = Document(page_content="Number of calories in 1 tbsp of garlic cream cheese is 80 calories")
    doc8 = Document(page_content="Number of calories in 1 tbsp of plain cream cheese is 60 calories")
    vector_db.add_documents([doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8])

    retriever = vector_db.as_retriever()
    # prompt = ChatPromptTemplate.from_messages([
    #   ("system", "YOu are a helpful assistant.")
    #   ("user", ".... {some_value}")
    # ])
    prompt = hub.pull("rlm/rag-prompt")
    # prompt.pretty_print()
    llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm.with_structured_output(FinalResponse)
    )

    final_list = []
    for item in response1.items:
        response = rag_chain.invoke(f"what is the number of calories in {item} ?")
        final_list.append(response)

    return final_list


def get_updated_calories(foods: list):
    from langchain_openai import OpenAIEmbeddings
    from langchain import hub

    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser

    embeddings = OpenAIEmbeddings(model="text-embedding-3-large",
                                  api_key="sk-nv-visualinsightagent-id-tuk8RSVmFbhC3Neo4EihT3BlbkFJzXcJWBrvUyYkNIacTwFt")
    connection = {"host": "127.0.0.1", "port": 19530}
    vector_db = Milvus(embedding_function=embeddings, connection_args=connection,
                       collection_name="calories",
                       auto_id=True,
                       drop_old=True)

    doc1 = Document(page_content="Number of calories in plain bagel is 180 calories")
    doc2 = Document(page_content="Number of calories in brown bread bagel is 150 calories")
    doc3 = Document(page_content="Number of calories in jalapeno cheese bagel is 230 calories")
    doc4 = Document(page_content="Number of calories in everything bagel is 200 calories")
    doc5 = Document(page_content="Number of calories in 1 tbsp of cream cheese is 70 calories")
    doc6 = Document(page_content="Number of calories in 1 tbsp of tomato basil cheese is 100 calories")
    doc7 = Document(page_content="Number of calories in 1 tbsp of garlic cream cheese is 80 calories")
    doc8 = Document(page_content="Number of calories in 1 tbsp of plain cream cheese is 60 calories")
    vector_db.add_documents([doc1, doc2, doc3, doc4, doc5, doc6, doc7, doc8])
    retriever = vector_db.as_retriever()
    # prompt = ChatPromptTemplate.from_messages([
    #   ("system", "YOu are a helpful assistant.")
    #   ("user", ".... {some_value}")
    # ])
    prompt = hub.pull("rlm/rag-prompt")
    # prompt.pretty_print()
    llm = ChatFireworks(model="accounts/fireworks/models/llama-v3p1-70b-instruct")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )

    final_response = []
    for food in foods:
        response = rag_chain.invoke(f"what is the number of calories in {food} ?")
        final_response.append(response)
    return final_response

# Function to handle image upload and generate a detailed bot response
def handle_image_upload(uploaded_image):
    # The food details list provided

    food_details = vlm_output(image_path="bagel.jpg")

    # food_details = [
    #     {
    #         "food": "Bagel",
    #         "calories": 70,
    #         "options": ["plain bagel", "sesame bagel", "raspberry bagel"]
    #     },
    #     {
    #         "food": "Cheese",
    #         "calories": 100,
    #         "options": ["cream cheese", "sour cheese", "cheddar cheese"]
    #     }
    # ]

    # Constructing the bot's initial response
    identified_foods = [item.food for item in food_details]
    bot_message = f"I identified the food to be {', '.join(identified_foods)}. Below is a more detailed breakdown of the diet. Please feel free to change to the correct option if I have selected the wrong option."

    return bot_message, food_details

# Function to render food options table
def render_food_options_table(food_details):
    for item in food_details:
        col1, col2, col3 = st.columns([1, 1, 2])
        col1.write(item.food)
        col2.write(item.calories)
        col3.selectbox("Select an option", item.food_options, key=item.food)

    if st.button("Update"):
        return get_updated_calories(["jalapeno cheese bagel", "tomato basil cheese"])
    return None


# Routing logic based on session state instead of query params
if "page" not in st.session_state:
    st.session_state["page"] = "signup"

if st.session_state["page"] == "signup":
    # Signup Form Page
    col1, col2 = st.columns([1, 6])
    with col1:
        st.image(icon, width=70)
    with col2:
        st.markdown("<h1 style='font-size: 40px; margin-bottom: -5px;'>Meal Maven Signup</h1>", unsafe_allow_html=True)

    st.markdown("---")

    with st.form(key='signup_form'):
        st.markdown("### User Information")
        username = st.text_input("User Name", placeholder="Enter your full name")
        email = st.text_input("User Email", placeholder="Enter your email address")

        st.markdown("### Personal Details")
        gender = st.radio("Gender", ('Male', 'Female', 'Other'), horizontal=True)
        age = st.slider("Age", min_value=10, max_value=100, value=25)
        height = st.slider("Height (in cm)", min_value=100, max_value=250, value=170)
        weight = st.slider("Weight (in kg)", min_value=30, max_value=200, value=70)

        st.markdown("### Fitness Goals")
        fitness_goal = st.radio("Fitness Goal", ('Weight Loss', 'Muscle Gain'), horizontal=True)

        # Dynamically update fitness range based on fitness goal
        fitness_range = st.radio("Fitness Range", ('5% change', '10% change', '12% change'), horizontal=True)

        fitness_timeline = st.slider("Fitness Timeline (in months)", min_value=1, max_value=24, value=6)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            # Save the data to a JSON file
            user_data = {
                "username": username,
                "email": email,
                "gender": gender,
                "age": age,
                "height": height,
                "weight": weight,
                "fitness_goal": fitness_goal,
                "fitness_timeline": fitness_timeline,
                "fitness_range": fitness_range,
                "timestamp": datetime.now().isoformat()
            }
            save_to_json(user_data)

            # Change page to chat after submission
            st.session_state["page"] = "chat"
            st.session_state["initial_message_sent"] = False  # Reset the flag
            st.rerun()

if st.session_state["page"] == "chat":
    # Chatbot Interface Page

    # Initialize session state variables
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    if "initial_message_sent" not in st.session_state:
        st.session_state["initial_message_sent"] = False

    # Ensure the bot's initial message is added only once
    if not st.session_state["initial_message_sent"]:
        initial_message = generate_initial_bot_message()
        st.session_state["messages"].append({"sender": "Meal Maven", "message": initial_message})
        st.session_state["initial_message_sent"] = True

    def add_message(sender, message):
        st.session_state["messages"].append({"sender": sender, "message": message})

    # Display all messages
    for message in st.session_state["messages"]:
        if message["sender"] == "User":
            st.markdown(
                f"<div style='text-align: right; color: red;'><strong>{message['sender']}: </strong>{message['message']}</div>",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"<div style='text-align: left; color: green;'><strong>{message['sender']}: </strong>{message['message']}</div>",
                unsafe_allow_html=True)

    # User input
    user_input = st.text_input("You:", placeholder="Type a message...")

    # Image uploader
    uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_image and "image_processed" not in st.session_state:
        bot_message, food_details = handle_image_upload(uploaded_image)
        add_message("Meal Maven", bot_message)
        st.session_state["food_details"] = food_details
        st.session_state["image_processed"] = True
        st.rerun()

    if "food_details" in st.session_state:
        update_response = render_food_options_table(st.session_state["food_details"])
        if update_response:
            add_message("Meal Maven", update_response)
            st.rerun()

if st.button("Send"):
    if user_input:
        add_message("User", user_input)
        # Here, you can implement a response from Meal Maven or any other functionality
        add_message("Meal Maven", "I'm here to help with your fitness journey!")

    # Display images in chat
    for message in st.session_state["messages"]:
        if message.get("image"):
            st.image(message["image"], caption=f"{message['sender']} uploaded this image.")