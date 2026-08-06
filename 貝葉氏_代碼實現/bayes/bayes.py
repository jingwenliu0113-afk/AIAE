import numpy as np
import re 
import random
import os


def textParse(input_string):
    listofTokens = re.split(r'\W+',input_string)
    return [tok.lower() for tok in listofTokens if len(tok)>2]  # 修正這裡的條件

def creatVocablist(doclist):
    vocabSet = set([])
    for document in doclist:
        vocabSet = vocabSet|set(document)
    return list(vocabSet)

def setOfWord2Vec(vocablist,inputSet):
    returnVec  = [0]*len(vocablist)
    for word in inputSet:
        if word in vocablist:
            returnVec[vocablist.index(word)] = 1
    return returnVec
    
def trainNB(trainMat,trainClass):
    numTrainDocs = len(trainMat)
    numWords = len(trainMat[0])
    p1 = sum(trainClass)/float(numTrainDocs)
    p0Num = np.ones((numWords)) #做了平滑處理
    p1Num = np.ones((numWords)) #拉普拉斯平滑
    p0Denom = 2
    p1Denom = 2 #通常情況下都是設定成類別個數
    
    for i in range(numTrainDocs):
        if trainClass[i] == 1: #垃圾郵件
            p1Num += trainMat[i]
            p1Denom += sum(trainMat[i])
        else:
            p0Num += trainMat[i]
            p0Denom += sum(trainMat[i])
    p1Vec = np.log(p1Num/p1Denom)
    p0Vec = np.log(p0Num/p0Denom)
    return p0Vec,p1Vec,p1
    
def classifyNB(wordVec,p0Vec,p1Vec,p1_class):    
    p1 = np.log(p1_class) + sum(wordVec*p1Vec)
    p0 = np.log(1.0 - p1_class) + sum(wordVec*p0Vec)
    if p0>p1:
        return 0
    else:
        return 1

def spam():
    # 使用 bayes.py 所在目錄作為基礎路徑，避免依賴特定電腦的絕對路徑
    base_path = os.path.dirname(os.path.abspath(__file__))
    spam_path = os.path.join(base_path, "email", "spam")
    ham_path = os.path.join(base_path, "email", "ham")
    
    print(f"垃圾郵件路徑: {spam_path}")
    print(f"正常郵件路徑: {ham_path}")
    
    # 檢查路徑是否存在
    if not os.path.exists(spam_path):
        print(f"錯誤：找不到垃圾郵件資料夾 {spam_path}")
        return
    if not os.path.exists(ham_path):
        print(f"錯誤：找不到正常郵件資料夾 {ham_path}")
        return
    
    doclist = []
    classlist = []
    loaded_files = 0
    
    for i in range(1,26):
        # 處理垃圾郵件
        spam_file = os.path.join(spam_path, f"{i}.txt")
        try:
            # 嘗試不同編碼
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'latin1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(spam_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content:
                wordlist = textParse(content)
                if wordlist:  # 確保不是空的
                    doclist.append(wordlist)
                    classlist.append(1) #表示垃圾郵件
                    loaded_files += 1
                    print(f"✅ 載入垃圾郵件 {i}.txt")
                else:
                    print(f"⚠️  垃圾郵件 {i}.txt 是空的")
            else:
                print(f"❌ 無法讀取垃圾郵件 {i}.txt")
                
        except FileNotFoundError:
            print(f"❌ 找不到垃圾郵件檔案 {spam_file}")
        except Exception as e:
            print(f"❌ 讀取垃圾郵件 {i}.txt 時發生錯誤: {e}")
        
        # 處理正常郵件
        ham_file = os.path.join(ham_path, f"{i}.txt")
        try:
            # 嘗試不同編碼
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'latin1']
            content = None
            
            for encoding in encodings:
                try:
                    with open(ham_file, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content:
                wordlist = textParse(content)
                if wordlist:  # 確保不是空的
                    doclist.append(wordlist)
                    classlist.append(0) #表示正常郵件
                    loaded_files += 1
                    print(f"✅ 載入正常郵件 {i}.txt")
                else:
                    print(f"⚠️  正常郵件 {i}.txt 是空的")
            else:
                print(f"❌ 無法讀取正常郵件 {i}.txt")
                
        except FileNotFoundError:
            print(f"❌ 找不到正常郵件檔案 {ham_file}")
        except Exception as e:
            print(f"❌ 讀取正常郵件 {i}.txt 時發生錯誤: {e}")
    
    print(f"\n總共成功載入 {loaded_files} 個檔案")
    
    if len(doclist) == 0:
        print("❌ 沒有載入任何郵件檔案！請檢查檔案路徑和內容。")
        return
    
    spam_count = sum(classlist)
    ham_count = len(classlist) - spam_count
    print(f"📊 垃圾郵件: {spam_count} 封，正常郵件: {ham_count} 封")
        
    vocablist = creatVocablist(doclist)
    print(f"📝 詞彙表大小: {len(vocablist)}")
    
    trainSet = list(range(len(doclist)))  # 改為動態長度
    testSet = []
    
    # 選擇測試集（最多10個，但不超過總數的20%）
    test_size = min(10, max(1, len(doclist) // 5))
    
    for i in range(test_size):
        if len(trainSet) == 0:
            break
        randIndex = int(random.uniform(0,len(trainSet)))
        testSet.append(trainSet[randIndex])
        del (trainSet[randIndex])
    
    print(f"🎯 訓練集: {len(trainSet)} 個，測試集: {len(testSet)} 個")
    
    trainMat = []
    trainClass = []
    for docIndex in trainSet:
        trainMat.append(setOfWord2Vec(vocablist,doclist[docIndex]))
        trainClass.append(classlist[docIndex])
    
    if len(trainMat) == 0:
        print("❌ 沒有訓練資料！")
        return
        
    print("🔄 開始訓練模型...")
    p0Vec,p1Vec,p1 = trainNB(np.array(trainMat),np.array(trainClass))
    
    print("📊 測試結果:")
    errorCount = 0
    for i, docIndex in enumerate(testSet):
        wordVec = setOfWord2Vec(vocablist,doclist[docIndex])
        predicted = classifyNB(np.array(wordVec),p0Vec,p1Vec,p1)
        actual = classlist[docIndex]
        
        pred_label = "垃圾" if predicted else "正常"
        actual_label = "垃圾" if actual else "正常" 
        result = "✅" if predicted == actual else "❌"
        
        print(f"  測試 {i+1}: 預測={pred_label} | 實際={actual_label} | {result}")
        
        if predicted != actual:
            errorCount += 1
    
    accuracy = (len(testSet) - errorCount) / len(testSet) * 100 if len(testSet) > 0 else 0
    print(f'\n📈 在 {len(testSet)} 個測試樣本中，錯了：{errorCount} 個')
    print(f'準確率：{accuracy:.1f}%') 

if __name__ == '__main__':
    spam()