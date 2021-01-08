# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 13:18:02 2020

@author: 91844
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 12:58:41 2020

@author: 91844
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:35:46 2020

@author: 91844
"""

from tkinter import *
from modelfitting import predict_price
root=Tk()

root.title('House Price Predictor')
root.geometry('1200x780')

image=PhotoImage(file='D:\\SDP(Sem3)\\rsz_house1.png')
label=Label(image=image)
label.pack()


l=Label(root,relief='ridge',text='House Price Predictor',font=('Times New Roman',40,'bold'),bg='gold')
l.place(x=350,y=5)

l1=Label(root,text='Enter Area (In w)',font=('Times New Roman',18),bg='khaki')
l1.place(x=305,y=150)

l2=Label(root,text='Enter Desired Location',font=('Times New Roman',18),bg='khaki')
l2.place(x=305,y=185)

l3=Label(root,text='BHK',font=('Times New Roman',18),bg='khaki')
l3.place(x=305,y=220)

l4=Label(root,text='Number of Bathrooms',font=('Times New Roman',18),bg='khaki')
l4.place(x=305,y=255)


t1=Entry(root,width=50)
t1.place(x=585,y=160)

t2=Entry(root,width=50)
t2.place(x=585,y=195)

t3=Entry(root,width=50)
t3.place(x=585,y=230)

t4=Entry(root,width=50)
t4.place(x=585,y=265)

def price_fun():
    
    a=t2.get()
    b=t1.get()
    c=t3.get()
    d=t4.get()
    
    
    try:
        a1=str(a)
        b1=float(b)
        c1=int(c)
        d1=int(d)
        
        price=predict_price(a1,b1,c1,d1)
        
        l6=Label(root,text=("{:.2f}".format(price),'Lakhs'),font=('Times New Roman',25),bg='light green')
        l6.place(x=655,y=395)
        
        # l7=Label(root,text='Lakhs',font=('Times New Roman',25),bg='khaki')
        # l7.place(x=850,y=395)
    except:
        l10=Label(root,text="Invalid Input",font=('Times New Roman',25),bg='khaki')
        l10.place(x=655,y=395)



b1=Button(root,text='Estimated Price',font=('Times New Roman',20),bg='orange red',command=price_fun)
b1.place(x=305,y=395)

def clearlabels():
    t1.delete(0,END)
    t2.delete(0,END)
    t3.delete(0,END)
    t4.delete(0,END)
    l6.destroy()
    # try:
    #     l3.destroy()
    # except:
    #     print('ok')
        
b4=Button(root,text='OK',font=('Times New Roman',20),bg='red',command=clearlabels)
b4.place(x=305,y=530)    

def quit():
    root.destroy()

b2=Button(root,text='Quit',font=('Times New Roman',20),bg='red',command=quit)
b2.place(x=305,y=630)




root.mainloop()