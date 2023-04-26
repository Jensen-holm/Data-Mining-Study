#!/bin/bash
npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css
gunicorn --bind :3000 --workers 1 --threads 8 --timeout 0 app:app