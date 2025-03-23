from flask import Flask, render_template
import os
app = Flask(__name__)
import pickle
# import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')

clf=pickle.load(file)
file.close()

# Check the current working directory
print("Current working directory:", os.getcwd())

# Check the contents of the 'templates' directory
templates_path = os.path.join(os.getcwd(), 'templates')
print("Templates directory path:", templates_path)
if os.path.exists(templates_path):
    print("Contents of 'templates' directory:", os.listdir(templates_path))
else:
    print("Templates directory does not exist.")

@app.route('/')
def hello_world():
    # code for infernce
    inputfeatures = [100,1,22,1,1]
    inf_prob=clf.predict_proba([inputfeatures])[0][1]# predict only preddict yes or no but probability tells abut the percentage 
    # return render_template("index.html")
    # return'helloworld'+str(inf_prob)
    return "index.html"
    
 
if __name__ == "__main__":
     app.run(debug=True)
     