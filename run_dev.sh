#!/bin/bash

# Function to handle SIGINT signal
function cleanup {
  echo "Cleaning up and exiting..."
  kill $TAILWIND_PID
  kill $FLASK_PID
  exit 0
}

# Register cleanup function to be called when SIGINT is received
trap cleanup SIGINT

# Start tailwindcss watch process in the background
npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css --watch &
TAILWIND_PID=$!

# Wait for the CSS files to be generated before starting the Flask server
sleep 2

# Start the Flask server in the background
python app.py &
FLASK_PID=$!

# Wait for both processes to finish
wait $TAILWIND_PID $FLASK_PID

# When the wait command completes, kill both processes
kill $TAILWIND_PID
kill $FLASK_PID
