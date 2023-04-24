## Flask-Tailwindcss-Template

### Preparing for your project
1. `git clone https://github.com/Jensen-holm/Flask-Tailwindcss-Template && cd Flask-Tailwindcss-Template` <br>
2. `rm -rf .git` <br>
3. `mv * ..` <br>
4. `cd ..` <br>
5. `rm -rf Flask-Tailwindcss-Template` <br>

### Develop
1. `pip3 install -r requirements.txt` <br>
2. `npm install` <br>
(in one terminal window)
3. `npx tailwindcss -i ./static/src/input.css -o ./static/dist/css/output.css --watch` <br>
<br>
(in another terminal window) <br>
4. `python3 app.py` <br>

### Prod
1. `chmod +x run.sh` <br>
2.  `./run.sh` <br>

