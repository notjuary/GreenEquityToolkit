
# GreenEquity Toolkit - Bachelor's thesis project in Computer Science

Implementation of analytical software with the goal of visualizing and obtaining aspects to social, environmental sustainability in machine learning systems.A web-app was realised using the high-level Python framework Django. CodeCarbon was used to assess environmental sustainability, while AIF360 was used for social sustainability. For the identification of pre-trained models, the open-source platform HuggingFace was used.

Would you like to download the repository locally and run it? Follow these steps:


## Run Locally

Clone the project

```bash
   git clone [https://github.com/notjuary/tesiSostenibilita]
```

Go to the cloned folder

```bash
   cd my-project
```

For the initial setup (Python version higher than 3.9.0 is required):

```bash
   python -m venv env
   env\Scripts\activate.bat
   pip install -r requirements.txt
   python manage.py migrate
   python manage.py runserver 
```

To run:

```bash
   env\Scripts\activate.bat
   python manage.py runserver
```
