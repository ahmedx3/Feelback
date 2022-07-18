# FeelBack: An Automated video feedback framework

# Steps to Run
- `conda env create -f environment.yml`
- Create a .env file like the example.env file and change the following variables: UPLOAD_FOLDER, ANNOTATED_UPLOAD_FOLDER, DATABASE_URL, THUMBNAILS_FOLDER and FLASK_RUN_PORT
- run `python feelback_server.py`
- cd into feelback-front and run `npm run serve`

# Update Conda Environment
- `conda activate feelback `
- `conda env update --file requirements.yml --prune`
