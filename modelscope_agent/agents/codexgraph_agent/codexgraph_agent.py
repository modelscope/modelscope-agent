from modelscope_agent.agents.codexgraph_agent import CodexGraphAgentGeneral

class CodexGraphAgentCommenter(CodexGraphAgentGeneral):
    def set_action_type_and_message(self):
        self.action_type = 'ADD_COMMENTS'
        self.generate_message = 'You are ready to add code comments.'

class CodexGraphAgentGenerator(CodexGraphAgentGeneral):
    def set_action_type_and_message(self):
        self.action_type = 'GENERATE_NEW_CODE'
        self.generate_message = 'You are ready to do generate New Code.'

class CodexGraphAgentUnitTester(CodexGraphAgentGeneral):
    def set_action_type_and_message(self):
        self.action_type = 'GENERATE_UNITTEST'
        self.generate_message = 'You are ready to do generate Professional Unittest.'