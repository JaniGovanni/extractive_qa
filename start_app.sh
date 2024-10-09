#!/bin/bash

# Start the API server in model_server (assuming it's a Python script named server.py)
cd model_server
python app.py &

# Wait for a few seconds to ensure the server is up
sleep 2

# Change back to the root directory
cd ..

# Run the Streamlit application
streamlit run app_website_chat.py

# run with: ./start_app.sh