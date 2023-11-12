class memory:
    def __init__(self,user_template: str = '',assistant_template: str = '',):
        self.user_template = user_template
        self.assistant_template = assistant_template
        self.reset()

    def reset(self):
        self.history = []
        
    def store(self,task):
        # store history
        self.history.append({
            'role':
            'user',
            'content':
            self.user_template.replace('<user_input>', task)
        })
        self.history.append({
            'role': 'assistant',
            'content': self.assistant_template
        })
        return self.history