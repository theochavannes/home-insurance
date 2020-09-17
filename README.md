Solution to the home insurance problem

More explanations about the shap library
can be found at this link: 

https://github.com/slundberg/shap

More explanations about the aikit library
can be found at these links:

https://aikit.readthedocs.io/en/latest/auto_ml.html

https://github.com/societe-generale/aikit


### Step 1: Create a virtual environment
Please create a virtual environment on which you will work with
this library. 

This can be done this way (in the command prompt):

conda create --name myenv

### Step 2: Activate your virtual environment
In the command prompt:

activate myenv

### Step 3: Install the library.
Go in the folder of the project (containing "setup.py"). 
Then run the following (in the command prompt):

pip install -e .

This will install the library home-insurance and all
the required libraries (in requirements.txt)
If you work on an OSX system, virtual envs and matplotlib
may failed to work together. More information about
how to use matplotlib in an OSX system and a virtual 
environment:

https://matplotlib.org/3.1.0/faq/osx_framework.html

### Step 4: Create a folder in which you will save all the outputs
Then, create an environment variable called
"HOME_INSURANCE_DATA_FOLDER" and set it
to the path of the folder in which you will 
save all the outputs.

You can also, instead, go in the file "config.py"
of the library, and replace
"C:\\data\\esure\\home-insurance" by your path.

### Step 5: Train the Predictor
In home_insurance/helpers, execute the file "train_model.py"
This will create a folder for this execution in the
path you specified, and it will store all the
relevant outputs (pickle of the Predictor, 
dependence plots etc.)

### Step 6: Try the predictor (API simulation)
In the folder home_insurance/helpers, in file
try_model.py, specify the model id of the model
you trained.
(change the value of the variable "model_name")

Run the file. This will simulate having a json 
to load (format of the input data). The data
should be stored in the "records" standard format. 
Then we just need to load it and 
create a DataFrame from it. The predictor is called
on it and outputs the predictions with
the explanations (i.e. the shap values).


### Step 7 (Optional): Run an automl
Once steps 1-5 are done, you can run an automl if you
want. Go in the folder home_insurance/helpers. Then 
in a prompt, execute:
python run_automl.py run

This will create the AUTOML_FOLDER in your path.
It will store in it all the required information.

Once you are ready to get the results, just execute
in a prompt:

python run_automl.py result

It will create an excel file result.xlsx
in the AUTOML_FOLDER.
You can then compare the performances of the different models.



