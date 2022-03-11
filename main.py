from tkinter import *
import numpy as np
import cv2
import torch
from model import Net
import random
def run_inference_on_frame(frame, model):
    res = cv2.resize(frame, dsize=(28, 28), interpolation = cv2.INTER_CUBIC)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    res1 = np.reshape(res, (1, 1, 28, 28)) / 255
    res1 = torch.from_numpy(res1)
    res1 = res1.type(torch.FloatTensor)
    out = model(res1)
    probs, label = torch.topk(out, 25)
    probs = torch.nn.functional.softmax(probs, 1)
    pred = out.max(1, keepdim=True)[1]
    return(pred, probs)
def clicked():
    cap = cv2.VideoCapture(0)
    font =cv2.FONT_HERSHEY_PLAIN

    cap.set(3, 700)
    cap.set(4, 480)

    modelo = torch.load('model_weights.pt')
    modelo.eval()

    signs = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I','9':'J',
            '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R',
            '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y' , '25':'Z'}
    
    prompt = 11
    score=0
    i = 0
    while True:
        ret, frame = cap.read()
        
        img = frame[20:250, 20:250]
        frame = cv2.rectangle(frame, (20, 20), (250, 250), (0, 255, 0), 3)
        frame = cv2.putText(frame, "Target: "+list(signs.values())[prompt], (60,385), font, 1, (255,255,0), 2, cv2.LINE_AA)
        frame = cv2.putText(frame, "Current Score: "+str(score), (60,355), font, 1, (255,0,0), 2, cv2.LINE_AA)
        k = cv2. waitKey(1)
        k=32
        if k%256 == 32:
            #space - run inference
            pred, probs =run_inference_on_frame(img, modelo)
        
            if float(probs[0,0]) < 0.0:
                text_ = 'Sign not detected'
            else:
                text_ = signs[str(int(pred))] + ': ' + '{:.2f}'.format(float(probs[0,0])*100) + '%'

            font = cv2.FONT_HERSHEY_SIMPLEX
            frame = cv2.putText(frame, text_, (60,285), font, 1, (255,255,255), 2, cv2.LINE_AA)
            
            cv2.imshow('Cam', frame)
            #print(text_, "index:", pred.item(), prompt)

            l=[11,22,24,2,1,20,18,14]
            if pred.item() == prompt:
                prompt = l[i]
                score+=1
                i+=1

        elif k%256 == 27:
            #escape - exit
            break
            
        else:
            cv2.imshow('Cam', frame)

    cap.release()
    cv2.destroyAllWindows()


window = Tk()

window.title("Game-ON")

window.geometry('1000x800')


btn = Button(window, text="Start Playing",command=clicked)
btn.grid(column=2, row=0)
btn.place(x=450, y=100)
window.mainloop()
