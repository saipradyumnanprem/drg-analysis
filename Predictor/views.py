from django.shortcuts import render
from django.http import HttpResponse
from django.template import loader

#machine learning modules
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from collections import Counter

def predictor(request):
  return render(request, "index.html")

def svcalgo(druguse):
    data=pd.read_csv('./Predictor/data.csv', encoding='windows-1252')

    data['Age'] = data['Age'].astype(int)
    data['Sex'] = data['Sex'].astype(int)
    data['Race'] = data['Race'].astype(int)
    data['DeathCity'] = data['DeathCity'].astype(int)
    data['DrugUsed'] = data['DrugUsed'].astype(int)

    train = [0.5, 0.6, 0.7, 0.8, 0.9]
    test = [0.5, 0.4, 0.3, 0.2, 0.1]

    ans = []

    for i in range(5):
        train_data, test_data = train_test_split(data, train_size = train[i], 
                                                test_size = test[i],
                                                random_state = 10)
        
        X=train_data.copy()
        Y=train_data.copy()

        #Remove unnecessary columns for input variable
        remove = ['DrugUsed']
        X.drop(remove, inplace =True, axis =1)

        #Remove unnecessary columns for output variable
        remove1=['Age','Sex','Race','DeathCity']
        Y.drop(remove1, inplace =True, axis =1)
      
        X= X.to_numpy()
        Y=Y['DrugUsed'].to_numpy()
        
        #SVC Classification
        
        clf =make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(X, Y)


        #Splitting test data
        X=test_data.copy()
        Y=test_data.copy()
        remove = ['DrugUsed']
        X.drop(remove, inplace =True, axis =1)
        remove1=['Age','Sex','Race','DeathCity']
        Y.drop(remove1, inplace =True, axis =1)

        print("For {} - {} Training data split: ".format(train[i] * 100, test[i] * 100))

        X= X.to_numpy()
        Y=Y['DrugUsed'].to_numpy()

        #Predicting the accuracy

        ypred = clf.predict(X)
        score=clf.score(X,Y)
        print("Score: "+str(score))

        #Sample Input
        #age=39    #Age
        #sex=0     #Male
        #race=0    #White
        #city=140  #OXFORD

        drug=['Amphet','Fentanyl_Analogue','Oxycodone','Tramad','Methadone',
              'Hydrocodone','Coacaine','Heroin','Hydromorphone','Benzodiazepine',
              'Oxymorphone']

        #Predicting the final answer
        sample_output=clf.predict([[druguse['age'],druguse['sex'],druguse['race'],druguse['city']]])[0]
        #print(drug[sample_output-1] + "\n") #Predicted answer
        ans.append(sample_output-1)

    return ans

def results(request):

      if request.method == 'POST': 
          age = request.POST['age']
          sex = request.POST['gender']
          race = request.POST['race']
          city = request.POST['deathcity']

          druguse = {'age': age, 'sex':sex, 'race':race, 'city':city}

      drug=['Amphet','Fentanyl Analogue','Oxycodone','Tramad','Methadone',
              'Hydrocodone','Coacaine','Heroin','Hydromorphone','Benzodiazepine',
              'Oxymorphone']

      ans = svcalgo(druguse) 

      def max_occ(List):
          occurence_count = Counter(List)
          return occurence_count.most_common(1)[0][0]
        
      drug_max_occ = max_occ(ans)
      #print("The predicted drug for the given parameters is: {}".format(drug[drug_max_occ]))

      answer = drug[drug_max_occ]
      
      print(answer)

      #return HttpResponse(answer) 
      return render(request, "results.html", context = { 'answer' : answer})
