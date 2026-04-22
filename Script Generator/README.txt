AI SCRIPT GENERATOR – SETUP & USAGE GUIDE

This project generates short dialogue-based scenes using AI. It uses:

* Ollama + Llama for story generation
* Happy Transformer for optional text cleanup
* A Python GUI (Tkinter) for user interaction

==============================

1. REQUIREMENTS
   ==============================

Before running the project, make sure you have:

* Python 3.9 or higher installed
* Internet connection (for downloading models)
* Basic command prompt knowledge

==============================
2. INSTALL OLLAMA (LLAMA MODEL RUNTIME)
=======================================

Ollama is used to run the Llama AI model locally.

Step 1 — Download Ollama

1. Open Google
2. Search for: "Ollama download"
3. Open the official website: https://ollama.com
4. Download the version for your OS (Windows/Mac/Linux)
5. Install it like a normal application

Step 2 — Verify Ollama Installation

Open Command Prompt and run:

ollama --version

If installed correctly, it will show a version number.

Step 3 — Download the Llama Model

Run this in Command Prompt:

ollama pull llama3

This downloads the AI model used in the project.

==============================
3. CREATE A VIRTUAL ENVIRONMENT
===============================

A virtual environment keeps your project dependencies separate.

Step 1 — Navigate to your project folder

cd "C:\Users\aidan\OneDrive\Desktop\Script Generator"

Step 2 — Create the virtual environment

python -m venv venv

This creates a folder named "venv".

==============================
4. ACTIVATE THE VIRTUAL ENVIRONMENT
===================================

On Windows (Command Prompt):

venv\Scripts\activate.bat

You should see:

(venv) C:\Users...

==============================
5. INSTALL REQUIRED LIBRARIES
=============================

Run this inside the activated virtual environment:

pip install ollama happytransformer transformers torch sentencepiece

(Optional) Save dependencies:

pip freeze > requirements.txt

==============================
6. RUN THE PROGRAM
==================

For GUI version:

python story_scene_generator_gui.py

For terminal version:

python story_scene_generator.py

==============================
7. HOW TO USE THE PROGRAM
=========================

1. Enter a story sentence
2. Enter a theme
3. Choose number of characters (1–5)
4. Enter duration (0–5 minutes)
5. Click "Generate Scene"

The system will:

* Generate characters
* Create dialogue
* Estimate timing
* Output structured JSON

==============================
8. SAVING OUTPUT
================

* Click "Save JSON" in the GUI
* Or export manually from terminal output

==============================
9. TROUBLESHOOTING
==================

Ollama not working:

Make sure Ollama is running:

ollama list

Model not found:

Run again:

ollama pull llama3

Module not found error:

Make sure virtual environment is activated:

(venv) ...

If not:

venv\Scripts\activate.bat
