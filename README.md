Solution to the home insurance problem

More explanations about the shap library
can be found at this link: 

https://github.com/slundberg/shap

More explanations about the aikit library
can be found at these links:

https://aikit.readthedocs.io/en/latest/auto_ml.html

https://github.com/societe-generale/aikit

The slides for the presentation are available inside the project:
Home insurance dataset case study.pptx

### Step 1: Create a virtual environment (optional)
Please create a virtual environment on which you will work with
this package. 

This can be done this way (in the command prompt):

`conda create --name myenv`

You do not need to create a virtual environment, but it is best
practice since the package has some packages requirements
(some specified version of scikit learn etc.). If you
do not use a virtual environment, it will replace your version
of your packages by the one specified in requirements.txt.

### Step 2: Activate your virtual environment (optional)
In the command prompt:

`activate myenv`

### Step 3: Install the package.
Go in the folder of the project (containing "setup.py"). 
Then run the following (in the command prompt):

`pip install -e .`

This will install the package home-insurance and all
the required packages (in requirements.txt)
If you work on an OSX system, virtual envs and matplotlib
may failed to work together. More information about
how to use matplotlib in an OSX system and a virtual 
environment:

https://matplotlib.org/3.1.0/faq/osx_framework.html

### Step 4: Create a folder in which you will save all the outputs
Then, create an (temporary) environment variable called
"HOME_INSURANCE_DATA_FOLDER" and set it
to the path of the folder in which you will 
save all the outputs. You can also, instead, go in the file "config.py"
of the package, and replace
"C:\\data\\esure\\home-insurance" by your path.

### Step 5: Do some data analysis
You can (optional) read the exploration.ipynb at the root of the project
which is a draft that allowed me
to start the project and get some ideas.

In home_insurance/helpers, run the data_analysis.py file. It 
will automatically create a folder data_analysis in your path,
and it will save various plots. Looking at these plots can give some
ideas of how the data looks like, and what variables initially
seems to be useful to predict the "lapsed" cases.


### Step 6: Train the Predictor
In home_insurance/helpers, execute the file "train_predictor.py"
This will create a folder for this execution in the
path you specified, and it will store all the
relevant outputs (pickle of the Predictor, 
dependence plots etc.)

### Step 7: Try the predictor (API simulation)
In the folder home_insurance/helpers, in file
try_model.py, specify the model id of the model
you trained.
(change the value of the variable "model_id")

Run the file. This will simulate having a json 
to load (format of the input data). The data
should be stored in the "records" standard format. 
Then we just need to load it and 
create a DataFrame from it. The predictor is called
on it and outputs the predictions with
the explanations (i.e. the shap values).


### Step 8 (Optional): Run an automl
Once steps 1-5 are done, you can run an automl if you
want. Go in the folder home_insurance/helpers. Then 
in a prompt, execute:

`python run_automl.py run`

This will create the AUTOML_FOLDER in your path.
It will store in it all the required information.

Once you are ready to get the results, just execute
in a prompt:

`python run_automl.py result`

It will create an excel file result.xlsx
in the AUTOML_FOLDER.
You can then compare the performances of the different models.



