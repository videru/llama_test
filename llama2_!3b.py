from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from PySide6 import QtCore, QtWidgets, QtGui
model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
# To use a different branch, change revision
# For example: revision="main"
model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                             device_map="auto",
                                             trust_remote_code=False,
                                             revision="main")

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)


class window(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Llama 2")
        self.resize(600, 400)
        self.layout = QtWidgets.QVBoxLayout()
        self.setLayout(self.layout)

        self.textbox = QtWidgets.QTextEdit()
        self.textbox.setPlaceholderText("Enter text here")
        self.textbox.setAcceptRichText(False)
        self.textbox.setTabChangesFocus(True)
        self.textbox.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.layout.addWidget(self.textbox)
        
        

        self.button = QtWidgets.QPushButton("Send")
        self.button.clicked.connect(self.send)
        self.layout.addWidget(self.button)
        
        self.textboxanswer = QtWidgets.QTextEdit()
        self.textboxanswer.setAcceptRichText(False)
        self.textboxanswer.setTabChangesFocus(True)
        self.textboxanswer.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        self.layout.addWidget(self.textboxanswer)

        self.textbox.setFocus()

    def send(self):
        self.textboxanswer.clear()
        prompt = self.textbox.toPlainText()
        prompt_template=f'''[INST] <<SYS>>
        You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
        <</SYS>>
        {prompt}[/INST]'''
        
        print("\n\n*** Generate:")



        input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.7, do_sample=True, top_p=0.95, top_k=40, max_new_tokens=512)
        print(tokenizer.decode(output[0]))

        # Inference can also be done using transformers' pipeline

        print("*** Pipeline:")
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=512,
            do_sample=True,#같은 질문에 대한 답변이 달라짐
            temperature=0.7,
            top_p=0.95,
            top_k=40,
            repetition_penalty=1.1
            , return_full_text=False #질문제거
        )

        #print(pipe(prompt_template)[0]['generated_text'])
        self.textboxanswer.setText(pipe(prompt_template)[0]['generated_text'])
        
    
app = QtWidgets.QApplication([])
window = window()
window.show()
app.exec_()



