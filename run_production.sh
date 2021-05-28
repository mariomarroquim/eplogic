killall python3

find -type d -exec chmod 750 {} \;
find -type f -exec chmod 640 {} \;

FLASK_ENV=production nohup python3 app.py > log.txt &
