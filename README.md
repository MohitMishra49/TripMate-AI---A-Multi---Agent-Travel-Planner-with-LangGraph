# ✈️ TripMate AI — A Multi-Agent Travel Planner with LangGraph

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-orange)
![MCP](https://img.shields.io/badge/MCP-Integrated-success)
![Render](https://img.shields.io/badge/Deployment-Render-46E3B7)
![License](https://img.shields.io/badge/License-MIT-green)

**🌐 Live Demo:** https://tripmate-ai-a-multi-agent-travel-planner-2xzv.onrender.com/

**📂 GitHub Repository:** https://github.com/MohitMishra49/TripMate-AI---A-Multi---Agent-Travel-Planner-with-LangGraph

---

## 📌 Overview

TripMate AI is a **Supervisor-based Multi-Agent Travel Planner** built using **LangGraph**, **Model Context Protocol (MCP)**, and **FastAPI**. The application demonstrates how multiple AI agents can collaborate to generate travel itineraries while maintaining safety through **guardrails** and **Human-in-the-Loop (HITL)** approval.

Instead of relying on a single AI model, TripMate AI uses a coordinated network of specialized agents managed by a supervisor agent. This architecture enables modular, scalable, and reliable AI workflows for complex travel planning tasks.

---

## 🚀 Features

* 🤖 Multi-Agent architecture powered by LangGraph
* 🎯 Supervisor Agent for intelligent task orchestration
* 🛡️ Input Guardrails for request validation
* 👤 Human-in-the-Loop (HITL) approval workflow
* 🌦️ MCP-based Weather Tool integration
* ⚡ FastAPI backend with REST APIs
* 💬 Interactive web interface
* ☁️ Live deployment on Render
* 🔄 Stateful travel planning workflow
* 🧩 Modular architecture for adding new AI agents

---

## 🏗️ System Architecture

```text
                    User
                      │
                      ▼
               Input Guardrails
                      │
                      ▼
              Supervisor Agent
                      │
      ┌───────────────┼────────────────┐
      │               │                │
      ▼               ▼                ▼
 Travel Planner   Weather Agent   Future MCP Tools
      │               │                │
      └───────────────┼────────────────┘
                      ▼
          Human-in-the-Loop Review
                      │
                      ▼
              Final Travel Plan
```

The **Supervisor Agent** coordinates the workflow, delegates tasks to specialized agents, gathers their responses, and produces a final itinerary while ensuring safety through guardrails and user approval.

---

## 🔄 Workflow

1. The user submits a travel request.
2. Input Guardrails validate and sanitize the request.
3. The Supervisor Agent analyzes the query.
4. The supervisor decides which specialized agents should be invoked.
5. MCP tools provide external information when required (e.g., weather).
6. The travel itinerary is generated.
7. The user can approve or request modifications through the Human-in-the-Loop process.
8. The finalized travel plan is returned.

---

## 🛠️ Tech Stack

### AI & Agent Frameworks

* LangGraph
* LangChain
* Model Context Protocol (MCP)

### Backend

* Python
* FastAPI
* Uvicorn

### Frontend

* HTML
* CSS
* JavaScript

### Deployment

* Render
* GitHub

---

## 📂 Project Structure

```text
TripMate-AI
│
├── app.py                         # FastAPI application and API routes
├── backend.py                     # Multi-agent orchestration logic
├── mcp_client.py                  # MCP client utilities
├── custom_weather_mcp_server.py   # Weather MCP server
├── static/                        # CSS, JS and frontend assets
├── templates/                     # HTML templates
├── requirements.txt
└── README.md
```

---

## 🔌 API Endpoints

| Method | Endpoint              | Description                                           |
| ------ | --------------------- | ----------------------------------------------------- |
| POST   | `/api/travel`         | Create or continue a travel planning conversation     |
| POST   | `/api/travel/approve` | Approve or request revisions to a generated itinerary |
| GET    | `/health`             | Application health check                              |

---

## 💡 Technical Highlights

* Supervisor-driven Multi-Agent architecture
* LangGraph state management
* MCP tool integration
* Human-in-the-Loop approval pipeline
* Input Guardrails for safer AI interactions
* Modular agent design
* RESTful FastAPI backend
* Cloud deployment using Render

---

## 📖 What I Learned

This project helped me gain hands-on experience with:

* Building production-style Multi-Agent AI systems
* Designing workflows using LangGraph
* Integrating external tools through MCP
* Implementing Human-in-the-Loop approval systems
* Applying AI safety using guardrails
* Building REST APIs with FastAPI
* Deploying AI applications on Render
* Designing modular and scalable AI architectures

---

## 🔮 Future Improvements

* ✈️ Flight search integration
* 🏨 Hotel booking APIs
* 💰 Budget optimization agent
* 🗺️ Interactive travel maps
* 📅 Calendar integration
* 🌍 Multi-language support
* 🧠 Long-term memory for travel preferences
* 📱 Mobile-responsive UI improvements
* 🤖 Additional MCP-powered travel tools

---

## 🤝 Contributing

Contributions are welcome!

If you have ideas for improvements, new MCP tools, or additional AI agents, feel free to open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the license included in this repository.

---

## 👨‍💻 Author

**Mohit Mishra**

* GitHub: https://github.com/MohitMishra49

If you found this project interesting, consider giving the repository a ⭐.
