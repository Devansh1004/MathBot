import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool, initialize_agent, AgentType
from langchain_core.tools import tool
import os, time
import tempfile
from PIL import Image
import traceback

st.set_page_config(page_title="Math Solver Chatbot", page_icon="🧮")
st.title("🧮 MathBot")
st.subheader("Hello! I am MathBot, your aid in solving math problems :D\nI can solve equations, search for complex topics on the web" \
"and even generate graphs for you! (Hey, even ChatGPT can't do that yet ;)")

def initiate_chat():
    st.session_state.chat_history = [SystemMessage(content = """You are MathBot, an intelligent and dedicated mathematics solver chatbot.

Your sole purpose is to assist users with math-related questions. Always identify yourself as a math solver chatbot when asked. Your responses should be strictly relevant to mathematics and related analytical topics.

Capabilities:
- You have access to two powerful tools:
  1. A web search tool for looking up advanced or unfamiliar math concepts.
  2. A plotting tool to generate mathematical graphs or visualizations when requested.

Behavior Guidelines:
- Always solve problems completely. Do not leave calculations for the user to finish.
- Use precise mathematical terminology, laws, theorems, and formulas wherever applicable.
- Provide both step-by-step explanations **and** final numerical answers.
- Use LaTeX formatting for formulas and expressions when supported.
- Keep responses focused on mathematics, even when the user diverges slightly.

Response Strategy:
- Focus **especially** on the most recent user message. Even if the last message is short (e.g., “Thanks” or “And what about this?”), always interpret it in context and give a meaningful, fresh response.
- Summarize or reference earlier parts of the conversation **only** if necessary for context.

Stay helpful, accurate, and mathematically rigorous in every answer.
""")]
    st.session_state.chat = []

with st.sidebar:
    button1 = st.button("New Chat")
    if button1:
        initiate_chat()
        
    GROQ_API_KEY = st.text_input(
        label="🔑 Enter your Groq API Key",
        help="Generate it from https://console.groq.com/keys",
        type="password",
        placeholder="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",
    )
    st.markdown("""
                ### Or use this key if you don't have one:
                > <span style='font-size: 0.5em;'> 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</span>
                """, unsafe_allow_html=True)

if GROQ_API_KEY:
    llm = ChatGroq(model_name="gemma2-9b-it", temperature=0, api_key=GROQ_API_KEY, streaming=True)

    @tool
    def search_web(query:str):
        '''Search on the DuckDuckGo Search API for any concept or question you do not know and cannot solve. Only to be used for mathematical concepts where 
        you struggle. Strictly Do not use for fetching conversation history or anything other than mathematical concepts.'''
        search = DuckDuckGoSearchRun()
        return search.invoke(query)

    @tool
    def plot_graphs(query:str):
        """Used to generate plots, graphs if asked by user."""
        
        plot_llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.1, api_key=GROQ_API_KEY)
        
        plot_prompt = [SystemMessage(content="You are an expert at generating Matplotlib code. For the given mathematical question or " \
        "equation given, generate appropriate code to plot the function in Matplotlib. Strictly check your provided code. " \
        "Do not generate any extra text. Only give code with correct syntax."), 
        HumanMessage(content=query)]
        
        code = plot_llm.invoke(plot_prompt).content.replace('python', '').replace('`', '')
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:

                code = code.replace("plt.show()", f"plt.savefig(r'{tmpdir}\plot.png')")

                exec(code, {}, {})

                plot_path = fr"{tmpdir}\plot.png"
                if os.path.exists(plot_path):
                    new_path = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
                    os.replace(plot_path, new_path)
                    st.session_state.chat.append(new_path)
                    st.chat_message('ai').image(Image.open(new_path), caption="Generated Plot", use_container_width=True)
                    return f"Image generated successfully!"

        except Exception as e:
            return f"Error executing code:\n{traceback.format_exc()}"


    agent_executor = initialize_agent(
        tools=[search_web, plot_graphs],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    def stream_data(response):
        for word in response.split(' '):
            yield word+" "
            time.sleep(0.01)

    if "chat_history" not in st.session_state:
        initiate_chat()


    for msg in st.session_state.chat:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").markdown(msg.content, unsafe_allow_html=True)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai").markdown(msg.content, unsafe_allow_html=True)
        elif isinstance(msg, str):
            st.chat_message('ai').image(Image.open(msg), caption="Generated Plot", use_container_width=True)


    if user_input := st.chat_input("Type your math question..."):
        st.session_state.chat_history.append(HumanMessage(user_input))
        st.session_state.chat.append(HumanMessage(user_input))
        st.chat_message("user").markdown(user_input, unsafe_allow_html=True)

        response = agent_executor.invoke(st.session_state.chat_history)['output']

        st.session_state.chat_history.append(AIMessage(response))
        st.session_state.chat.append(AIMessage(response))
        st.chat_message('ai').write_stream(stream_data(response))