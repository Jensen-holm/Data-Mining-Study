#!/bin/bash

gunicorn --bind :3000 --workers 1 --threads 8 --timeout 0 app:app
