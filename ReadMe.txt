1. create a virtual environment start anaconda prompt on windows

> create -n cisco

2. activate environment
> conda activate cisco

3. install requirements

> pip install -r requirements.txt

4. go to the directory where the project is

> python manage.py makemigrations
> python manage.py migrate

5. serve the project
python manage.py runserver