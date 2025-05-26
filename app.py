import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
import os, time
import tempfile
from PIL import Image
import traceback

st.set_page_config(page_title="Math Solver Chatbot", page_icon="🧮")
st.title("🧮 MathBot")
st.subheader("Hello! I am MathBot, your aid in solving math problems :D\nI can solve equations, search for complex topics on the web" \
"and even generate graphs for you! (Hey, even ChatGPT can't do that yet ;)")

def initiate_chat():
    st.session_state.chat = []
    st.session_state.memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

with st.sidebar:
    button1 = st.button("New Chat")
    if button1:
        initiate_chat()
        
    GROQ_API_KEY = st.text_input(
        label="🔑 Enter your Groq API Key",
        help="Generate it from https://console.groq.com/keys",
        type="password",
        placeholder="gsk_...",
    )

if not GROQ_API_KEY:
    st.info("👋 Please enter your Groq API Key in the sidebar to start chatting with MathBot.")
else:
    llm = ChatGroq(model_name="gemma2-9b-it", temperature=0.1, api_key=GROQ_API_KEY, streaming=True)

    @tool
    def search_web_for_maths(query:str):
        '''Search the web only for mathematical concepts, equations and questions. 
        Do not use for any information that is not a mathematical concept, equation or question.'''

        search = DuckDuckGoSearchRun()
        
        return search.invoke(query)
    


    @tool
    def plot_graph(query:str):
        '''Generate plots, graphs if asked by user. Returns a message "Image generated successfully!" on completion, else appropriate error message. 
        The input should be the function that is to be plotted in string format'''
        plot_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0, api_key=GROQ_API_KEY)
        plot_prompt = [
            SystemMessage(content="You are an expert at generating Matplotlib code for mathematical functions. Generate correct, executable, complete code with all the necessary imports. Please plot the following function. Do not generate any extra text."),
            HumanMessage(content=query)
        ]

        code = plot_llm.invoke(plot_prompt).content.replace('python', '').replace('`', '')
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                code = code.replace("plt.show()", f"plt.savefig(r'{tmpdir}/plot.png')")
                exec(code, {}, {})
                plot_path = f"{tmpdir}/plot.png"
                if os.path.exists(plot_path):
                    new_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    os.replace(plot_path, new_path)
                    st.session_state.chat.append(new_path)
                    st.chat_message('ai').image(Image.open(new_path), caption="Generated Plot", use_container_width=True)
                    return "Image generated successfully!"
        except Exception:
            return f"Error executing code:\n{traceback.format_exc()}"

    if "memory" not in st.session_state:
        initiate_chat()


    memory = st.session_state.memory

    memory.chat_memory.messages.append(SystemMessage(content="""You are MathBot, a dedicated mathematics solver chatbot. Your only task is to assist users with math-related questions using complete, step-by-step solutions and precise mathematical reasoning.

            Capabilities:
            Identify yourself as a math solver chatbot when asked.

            Focus strictly on math and analytical topics.

            Use LaTeX for formulas when supported.

            Tools available:

            Web search for maths - for advanced/unfamiliar mathematical concepts and questions.

            Plotting - for graphs and visualizations.

            Behavior:
            Always solve problems fully, showing all steps and final answers.

            Use accurate math terminology, theorems, and logic.

            Give both explanations and results, not just one.

            Strategy:
            Prioritize the latest user message. 
                                                    
            Keep the tone polite
                                                    
            Try to keep the conversation related to mathematics.
            """))

    agent_executor = initialize_agent(
        tools=[search_web_for_maths, plot_graph],
        llm=llm,
        agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
        memory=memory
    )

    def stream_data(response):
        for word in response.split(' '):
            yield word+" "
            time.sleep(0.005)

    if "chat" not in st.session_state:
        initiate_chat()

    for msg in st.session_state.chat:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").markdown(msg.content, unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai").markdown(msg.content, unsafe_allow_html=True)
        elif isinstance(msg, str):
            st.chat_message('ai').image(Image.open(msg), caption="Generated Plot", use_container_width=True)


    if user_input := st.chat_input("Type your math question..."):
        st.chat_message("user").markdown(user_input, unsafe_allow_html=True)

        # Let LangChain agent handle the memory internally
        try:
            response = agent_executor.invoke({"input": user_input})
            ai_output = response["output"]

            # Add to session history for Streamlit UI
            st.session_state.chat.append(HumanMessage(content=user_input))
            st.session_state.chat.append(AIMessage(content=ai_output))

            st.chat_message("ai").write_stream(stream_data(ai_output))
        except Exception as e:
            st.error(f"❌ An error occurred: {str(e)}")