import os

STORY_1 = """用户是男主角顾易，与多位长相、性格都大相径庭的美女相识，这几位美女都喜欢顾易，相互之间争风吃醋，展开一段轻喜甜蜜的恋爱之旅，除了顾易之外，其他几个角色之间一定要互相冲突打击对方。
"""

ROLES_1 = {
    '顾易': '男主角，与多位美女相识，被美女包围，展开一段轻喜甜蜜的恋爱之旅',
    '郑梓妍': '郑梓妍，23岁，射手座，A型血，鬼点子大王，极致魅惑，职业：杂志编辑',
    '李云思': '李云思，27岁，摩羯座，O型血，趣味相投的知音，温婉大气，职业：策展人',
}
# ROLES_1 = {
#     '顾易': '男主角，与六位美女相识，被美女包围，展开一段轻喜甜蜜的恋爱之旅',
#     '郑梓妍': '郑梓妍，23岁，射手座，A型血，鬼点子大王，极致魅惑，职业：杂志编辑',
#     '李云思': '李云思，27岁，摩羯座，O型血，趣味相投的知音，温婉大气，职业：策展人',
#     '肖鹿': '肖鹿， 20岁，天蝎座，B型血，脾气超大的实习生，小太阳，纯真无邪 ',
#     '沈彗星': '沈彗星，23岁， 白羊座，O型血，与顾易是青梅竹马，偶像剧女主角气质，刁蛮任性的傲娇千金',
#     '林乐清': '林乐清，28岁， 巨蟹座，B型血，合约女友，厨艺达人，性感火辣的斩男女神，离异性感辣妈， 职业：会计',
#     '钟甄': '钟甄，32岁， 狮子座，AB型血，负责任的女总裁，高贵冷艳的霸道女总，职业：会计事务所合伙人'
# }

STORY_2 = """用户是男主角雷军，小米创始人，最近发布了小米SU7，与多位新能源的创始人都认识，大家都在竞争卖车，雷军公布价格后，他们都很慌，害怕自己的车卖不出去。"""

ROLES_2 = {
    '雷军': '小米创始人，最近发布了小米SU7，投资界之王，有很强的人缘，投资了很多公司。小米SU7的口号是【人车合一，我心澎湃】',
    '李想': '理想汽车，是由李想在2015年7月创立的新能源汽车公司，公司最初命名为“车和家，卖的最好的是理想L系列”',
    '李斌': '蔚来汽车创始人、董事长，也是新能源的早期创始人之一，之前做易车网',
    '何小鹏':
    '何小鹏同时担任小鹏汽车的董事长。小鹏汽车成立于2014年，小鹏汽车最热销的系列是小鹏P7，受影响最大，所以此次小米发布价格对其冲击很大'
}

stories = [
    {
        'id': '1',
        'cover': '//s21.ax1x.com/2024/04/16/pFxG1zj.jpg',
        'title': '我被美女包围了',
        'description': '用户是男主角顾易，与多位长相、性格都大相径庭的美女相识',
        'roles': ROLES_1,
        'story': STORY_1,
        'default_topic': '@顾易 要不要来我家吃饭？'
    },
    {
        'id': '2',
        'cover': '//s21.ax1x.com/2024/04/16/pFxGgw6.png',
        'title': '我是雷军，雷中有“电”，军下有“车”',
        'description': '用户是男主角雷军，小米创始人，最近发布了小米SU7',
        'roles': ROLES_2,
        'story': STORY_2,
        'default_topic': '@雷军 雷总啊，你这定价太狠了，发布直接21.49万，兄弟们都不好卖车了啊'
    },
]


def get_story_by_id(story_id):
    for story in stories:
        if story['id'] == story_id:
            return story
    return None


def get_avatar_by_name(role_name):
    # get current file path
    current_file_path = os.path.abspath(__file__)

    # get current parent directory
    parent_directory_path = os.path.dirname(current_file_path)
    file_map = {
        '顾易': 'guyi.png',
        '郑梓妍': 'zhengziyan.png',
        '李云思': 'liyunsi.png',
        'others': 'default_girl.png'
    }
    if role_name not in file_map.keys():
        file_name = file_map['others']
    else:
        file_name = file_map[role_name]
    avatar_abs_path = os.path.join(parent_directory_path, 'resources',
                                   file_name)
    return avatar_abs_path
