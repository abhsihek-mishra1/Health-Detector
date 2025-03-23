from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')

clf=pickle.load(file)
file.close()

@app.route('/',methods=["GET","POST"])
def hello_world():
      
     if request.method == "POST":
          print(request.form)
          # myDict=request.form
          # try:
          #  Fever=int(myDict['fever'])
          #  Age=int(myDict['Age'])
          #  Pain=str(myDict['Body pain'])
          #  RunnyNose=int(myDict['Runny nose'])
          # except ValueError as e:
          #   return f"Input Error: {str(e)}"
          # try:
          #     DiffBredth =int(myDict['Diff Breathing'])
          # except ValueError as e:
          #   return f"Input Error: {str(e)}"
    
     #code for infernce
          # inputFeatures = [Fever,Age,Pain,RunnyNose,DiffBredth]
          inputFeatures = [102, 1, 22, 1, 1]
          inf_prob=clf.predict_proba([inputFeatures])[0][1]# predict only preddict yes or no but probability tells abut the percentage 
          print(inf_prob)
          return render_template("show.html",inf=round(inf_prob*100))
     return render_template("index.html")  
#     return'helloworld'+str(inf_prob)
 
 
if __name__ == "__main__":
     app.run(debug=True)

