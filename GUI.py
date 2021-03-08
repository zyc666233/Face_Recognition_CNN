import tkinter as tk

def start_GUI():
    window = tk.Tk()
    window.title('欢迎使用人脸识别系统')

    # window.geometry('500x500')

    # 这里是窗口的内容
    words = tk.StringVar()
    showpic = tk.Label(window, textvariable=words, font=(15), height=20)
    showpic.grid(row=1, column=0, columnspan = 2)

    def recognize():
        words.set('正在识别中...请稍后...')  # 设置标签的文字为 'you hit me'

    button1 = tk.Button(window, text="开始识别", width=25, height=4, command=recognize)
    button1.grid(row=0, column=0)
    button2 = tk.Button(window, text="录入数据", width=25, height=4, command=recognize)
    button2.grid(row=0, column=1)




    window.mainloop()

if __name__ == '__main__':
    start_GUI()