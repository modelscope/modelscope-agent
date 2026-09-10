export type Language = 'zh' | 'en';
export const repo = 'https://github.com/modelscope/ms-agent';
export const content = {
  zh: {
    lang: 'zh-CN',
    title: 'MS-Agent — 复杂任务 自主执行',
    description:
      '轻量、可组合、可扩展的智能体框架，以可定制的 Agent Harness 支持多智能体协作、长程自主执行与开放生态集成。',
    skip: '跳转到主要内容',
    nav: ['框架设计', '功能演示', '专业应用', '技能进化', '开放生态'],
    docs: '文档',
    menu: '打开导航',
    eyebrow: '轻量 · 可组合 · 可扩展',
    hero: ['复杂任务', '自主执行'],
    intro:
      '自由组合模型、工具、技能与子智能体，以可定制的 Harness 统筹上下文管理、权限控制、规划、执行与检查，构建你的专属生产力助手。',
    start: '开始构建',
    watch: '观看演示',
    tagline: 'A lightweight framework to empower agentic execution of complex tasks.',
    heroCaption: '上下文管理 · 权限控制 · 质量检查',
    parts: ['模型', '工具', '技能', '子智能体'],
    outputs: ['研究分析', '软件开发', '视频创作'],
    featureLine: ['模块化构建', '可定制 Harness', '长程自主执行'],
    composeEyebrow: '01 / COMPOSABLE AGENTS',
    composeTitle: '模块化构建与多智能体协作',
    composeIntro:
      '以配置组合角色、模型与工具，按需定制专业智能体。支持子任务委派与工作流编排，灵活组织复杂任务中的分工与协作。',
    roles: [
      {
        name: 'General',
        subtitle: '通用执行',
        desc: '任务统筹与自主委派',
        detail: '根据任务需要调度探索、开发和研究子智能体，统一管理进度与结果。',
      },
      {
        name: 'Plan',
        subtitle: '任务规划',
        desc: '需求分析与计划生成',
        detail: '梳理需求和项目资料，检查计划覆盖范围，为执行提供明确步骤。',
      },
      {
        name: 'Research',
        subtitle: '深度研究',
        desc: '信息检索与研究综合',
        detail: '交叉核对来源并组织研究结论，结合质量检查控制报告完整性。',
      },
      {
        name: 'Explore',
        subtitle: '只读探索',
        desc: '文件检索与信息定位',
        detail: '通过只读工具搜索和分析项目资料，为规划与执行提供依据。',
      },
      {
        name: 'Build',
        subtitle: '软件开发',
        desc: '代码实现与反馈修正',
        detail: '完成编写、运行和诊断，根据检查结果修复问题并验证实现。',
      },
    ],
    snippet: '配置节选',
    customize: '扩展接口',
    extensions: ['自定义工具', '生命周期 Hook', 'MCP 服务', 'Skills'],
    harnessEyebrow: '02 / COMPOSABLE HARNESS',
    harnessTitle: '自由定制\nAgent Harness',
    harnessIntro:
      '按需组合上下文管理、权限控制与质量检查，定制智能体的执行策略。通过生命周期扩展点接入业务规则，控制规划、协作与反馈过程。',
    steps: [
      {
        label: '计划',
        title: '任务分解与计划检查',
        text: '组织待办与执行步骤，检查计划是否覆盖用户要求，并注入当前运行环境。',
        code: 'callbacks:\n  - plan_check\n  - state_inject',
        chips: ['任务分解', '计划检查'],
      },
      {
        label: '行动',
        title: '工具调用与协作控制',
        text: '调用工具或委派子任务，通过权限、委派限制与重复调用检测管理执行过程。',
        code: 'callbacks:\n  - subagent_limit\n  - loop_guard',
        chips: ['工具调用', '子任务委派'],
      },
      {
        label: '检查',
        title: '质量检查与反馈修正',
        text: '检查产物、待办和回答质量，将发现的问题反馈给智能体继续修正。',
        code: 'callbacks:\n  - todo_gate\n  - stop_gate',
        chips: ['产物检查', '质量反馈'],
      },
      {
        label: '结束',
        title: '完成判定与预算管理',
        text: '通过结束条件、轮次提醒和重试限制，管理任务完成状态与执行预算。',
        code: 'round_reminder:\n  remind_before_max_round: 2\nstop_gate:\n  max_retries: 2',
        chips: ['轮次提醒', '结束条件'],
      },
    ],
    feedback: '反馈驱动修正',
    diagramNote: '规划 · 执行 · 检查 · 收敛',
    autonomyEyebrow: '03 / LONG-HORIZON EXECUTION',
    autonomyTitle: '长程自主执行',
    autonomyIntro: '结合自主调度、分层权限与项目记忆，支持多轮协作和周期任务，减少人工值守，保持任务连续性。',
    loopTitle: '自主调度与周期任务',
    loopText:
      '支持固定间隔与自主唤醒，按任务进展安排后续执行。通过轮次、时长与用量预算管理持续监测、巡检和维护任务。',
    loopNodes: ['执行任务', '调度唤醒', '继续执行'],
    loopFooter: 'Loop / Cron · 固定周期与自主调度',
    approvalTitle: '分层权限与自动审批',
    approvalText:
      '结合授权规则、模型评估与人工确认，按操作风险控制执行权限。常规请求自动审批，未能确定的请求交由人工处理。',
    approvalNodes: ['权限请求', '规则 / 委托评估', '允许一次', '人工确认'],
    approvalFooter: '规则优先 · 单次授权 · 人工兜底',
    continuityTitle: '上下文与状态管理',
    continuityItems: [
      ['上下文管理', '历史摘要与工具输出压缩'],
      ['项目记忆', '项目知识与工作偏好持久化'],
      ['会话恢复', '历史记录与执行过程回看'],
    ],
    demoEyebrow: '04 / BUILT-IN WEBUI',
    demoTitle: '开箱即用的智能体工作台',
    demoIntro: '统一管理项目、会话与文件，实时查看计划和工具调用，在交互中完成任务并交付成果。',
    demoPlay: '播放演示',
    demoPause: '暂停演示',
    demoExpand: '放大查看截图',
    demoClose: '关闭大图',
    demoFrames: ['工作计划', '执行与产物', '项目文件'],
    demoDetails: ['项目与会话管理', '执行过程可视化', '文件与成果预览'],
    demoCaption: 'WebUI 实录 · Qwen3.8-Max · 活动资料整理与简报生成',
    appsEyebrow: '05 / APPLICATIONS',
    appsTitle: '研究｜开发｜创作',
    appsIntro: '以专业智能体与工作流覆盖研究分析、软件开发和视频创作，支持领域定制与应用集成。',
    apps: [
      {
        id: 'research',
        name: '深度研究与文档分析',
        label: 'Deep Research / Doc Research',
        text: '组合检索、证据管理与报告生成，支持开放研究和文档分析。',
        tags: ['证据管理', '报告生成'],
        path: 'projects/deep_research/v2',
        art: ['问题', '检索 · 证据', '分析 · 综合', '研究报告'],
      },
      {
        id: 'code',
        name: '软件开发',
        label: 'CodeGenesis',
        text: '以需求、架构、编码和验证工作流完成项目开发。',
        tags: ['多阶段协作', '代码验证'],
        path: 'projects/code_genesis',
        art: ['需求', '架构', '编码', '验证'],
      },
      {
        id: 'finance',
        name: '金融研究',
        label: 'FinResearch',
        text: '整合金融数据、分析方法与报告规范，构建专业研究流程。',
        tags: ['金融数据', '专业分析'],
        path: 'projects/fin_research',
        art: ['市场资料', '数据收集', '专业分析', '研究报告'],
      },
      {
        id: 'cinema',
        name: '视频创作',
        label: 'Cinema',
        text: '编排脚本、图像、配音与视频工具，完成内容生成和后期合成。',
        tags: ['多模态生成', '视频合成'],
        path: 'projects/singularity_cinema',
        art: ['脚本', '图像', '音频', '成片'],
      },
    ],
    viewProject: '查看项目',
    applicationDiagram: '工作流程示意',
    evolutionTitle: '评估驱动的技能进化',
    evolutionText:
      '分析执行轨迹，结合评价与反思自动修订技能，通过任务验证筛选有效更新，持续改进智能体的执行方法。',
    evolutionSteps: ['执行', '评价', '反思', '修订', '验证'],
    connectEyebrow: '07 / OPEN ECOSYSTEM',
    connectTitle: '开放协议与跨框架生态',
    connectText:
      '兼容主流模型与社区插件，通过标准协议连接外部工具和智能体，复用不同框架中的技能、指令与记忆。',
    getEyebrow: '08 / GET STARTED',
    getTitle: '构建你的专属智能体',
    getIntro: '通过 WebUI 快速体验，在 CLI 中执行任务，或使用 Python SDK 集成到自己的应用。',
    getTabs: ['WebUI', 'CLI', 'Python SDK'],
    copy: '复制代码',
    copied: '已复制',
    copyFailed: '复制失败，请手动选择代码',
    installHelp: [
      '启动工作台，在模型设置中完成配置。',
      '配置模型后，直接从终端发起任务。',
      '使用配置定义智能体，通过 SDK 调用与扩展。',
    ],
    guide: '阅读使用文档',
    sdkExample: '查看完整 SDK 示例',
    footer: 'ModelScope 开源项目',
    footerLine: '面向复杂任务的轻量智能体框架',
    legal: 'Apache 2.0',
    back: '回到顶部',
    evolutionEyebrow: '06 / SKILL EVOLUTION',
    evolutionCaption: '任务反馈 → 技能改进',
    evolutionOutcome: '验证通过后保留更新',
    evolutionDetails: [
      ['轨迹分析', '定位执行中的问题'],
      ['技能修订', '生成候选技能方案'],
      ['验证筛选', '评估并保留有效改进'],
    ],
    modelLabel: '多模型接入',
    protocolCards: [
      {
        name: 'MCP',
        title: '工具与服务互联',
        text: '接入外部工具，也可将框架工具与专业应用发布为 MCP 服务。',
        icon: 'tool',
      },
      {
        name: 'ACP',
        title: '编辑器与智能体接入',
        text: '连接支持 ACP 的编辑器，调用外部智能体，并通过代理路由会话。',
        icon: 'terminal',
      },
      {
        name: 'A2A',
        title: '跨服务智能体协作',
        text: '调用远程智能体，或将 MS-Agent 发布为可发现、可调用的智能体服务。',
        icon: 'team',
      },
    ],
    pluginsTitle: '社区插件复用',
    pluginsText:
      '兼容 Claude Code、Codex、Cursor、OpenClaw 等插件格式，加载其中的技能、命令、Agent 定义、Hooks 与 MCP 配置。',
    pluginsFlow: ['社区插件', '组件加载', '智能体能力'],
    hubTitle: 'Agent Hub 跨框架迁移',
    hubText: '迁移、合并与同步指令、技能和记忆，连接 ModelScope Hub，复用跨框架的智能体资源。',
    hubResources: ['指令', '技能', '记忆'],
    ecosystemLink: '了解集成方式',
    models: ['Qwen', 'DeepSeek', 'GLM', 'Kimi', 'OpenAI', 'Anthropic'],
    pluginFormats: ['Claude Code', 'Codex', 'Cursor', 'OpenClaw'],
    hubFrameworks: ['QwenPaw', 'OpenClaw', 'Hermes', 'Qoder', 'Nanobot', 'OpenHuman'],
  },
  en: {
    lang: 'en',
    title: 'MS-Agent — A lightweight framework for complex tasks',
    description:
      'A lightweight framework to empower agentic execution of complex tasks. Compose models, tools, skills, and subagents with a customizable agent harness.',
    skip: 'Skip to content',
    nav: ['Design', 'Demo', 'Applications', 'Evolution', 'Ecosystem'],
    docs: 'Docs',
    menu: 'Open navigation',
    eyebrow: 'LIGHTWEIGHT · COMPOSABLE · EXTENSIBLE',
    hero: ['Complex tasks', 'Agentic execution'],
    intro:
      'Compose models, tools, skills, and subagents. Customize the harness for context, permissions, planning, execution, and verification to build your own productivity agent.',
    start: 'Start building',
    watch: 'Watch the demo',
    tagline: 'A lightweight framework to empower agentic execution of complex tasks.',
    heroCaption: 'Context · Permissions · Quality checks',
    parts: ['Models', 'Tools', 'Skills', 'Subagents'],
    outputs: ['Research', 'Development', 'Video creation'],
    featureLine: ['Modular agents', 'Customizable harness', 'Long-horizon execution'],
    composeEyebrow: '01 / COMPOSABLE AGENTS',
    composeTitle: 'Modular agents, flexible collaboration',
    composeIntro:
      'Configure roles, models, and tools for specialized agents. Combine task delegation with workflow orchestration to coordinate complex work.',
    roles: [
      {
        name: 'General',
        subtitle: 'Coordinate & execute',
        desc: 'Task coordination and delegation',
        detail:
          'Coordinate exploration, development, and research specialists while managing overall progress and results.',
      },
      {
        name: 'Plan',
        subtitle: 'Plan before acting',
        desc: 'Requirements analysis and planning',
        detail: 'Review requirements and project context, then check that the plan covers the task.',
      },
      {
        name: 'Research',
        subtitle: 'Investigate & synthesize',
        desc: 'Search and research synthesis',
        detail: 'Cross-check sources and develop grounded conclusions, with checks for report completeness.',
      },
      {
        name: 'Explore',
        subtitle: 'Read-only discovery',
        desc: 'File search and discovery',
        detail: 'Use read-only tools to locate and analyze information for planning and execution.',
      },
      {
        name: 'Build',
        subtitle: 'Implement & refine',
        desc: 'Implementation and verification',
        detail: 'Write, run, and diagnose code, then use feedback to refine and verify the result.',
      },
    ],
    snippet: 'Configuration excerpt',
    customize: 'Extension points',
    extensions: ['Custom tools', 'Lifecycle hooks', 'MCP servers', 'Skills'],
    harnessEyebrow: '02 / COMPOSABLE HARNESS',
    harnessTitle: 'Compose your\nagent harness',
    harnessIntro:
      'Combine context management, permissions, and quality checks to shape agent execution. Use lifecycle hooks to bring your own rules into planning, collaboration, and feedback.',
    steps: [
      {
        label: 'Plan',
        title: 'Make the task actionable',
        text: 'Organize multi-step work with todos and check that the plan is complete and feasible before acting.',
        code: 'callbacks:\n  - plan_check\n  - state_inject',
        chips: ['Task breakdown', 'Plan checks'],
      },
      {
        label: 'Act',
        title: 'Put tools and specialists to work',
        text: 'Call tools or delegate focused tasks. Tool boundaries and concurrency limits shape each execution step.',
        code: 'callbacks:\n  - subagent_limit\n  - loop_guard',
        chips: ['Tool calls', 'Delegation'],
      },
      {
        label: 'Check',
        title: 'Turn feedback into the next action',
        text: 'Inspect artifacts, pending todos, and answer quality. Return specific feedback so the agent can address what is missing.',
        code: 'callbacks:\n  - todo_gate\n  - stop_gate',
        chips: ['Artifact checks', 'Quality feedback'],
      },
      {
        label: 'Finish',
        title: 'Help complex work converge',
        text: 'Finish after checks pass. Round reminders, retry limits, and budgets help bring extended tasks to a close.',
        code: 'round_reminder:\n  remind_before_max_round: 2\nstop_gate:\n  max_retries: 2',
        chips: ['Round reminders', 'Stop conditions'],
      },
    ],
    feedback: 'Refine with feedback',
    diagramNote: 'Plan · Act · Check · Finish',
    autonomyEyebrow: '03 / LONG-HORIZON EXECUTION',
    autonomyTitle: 'Long-horizon autonomous execution',
    autonomyIntro:
      'Coordinate ongoing work with adaptive scheduling, layered permissions, and project memory, reducing manual supervision across multi-step and recurring tasks.',
    loopTitle: 'Adaptive and recurring execution',
    loopText:
      'Run at fixed intervals or let agents schedule their next wake-up. Manage monitoring and maintenance tasks with limits on rounds, duration, and usage.',
    loopNodes: ['Run task', 'Schedule wake-up', 'Continue'],
    loopFooter: 'Loop / Cron · Fixed intervals and adaptive scheduling',
    approvalTitle: 'Layered permissions and auto-approval',
    approvalText:
      'Combine permission rules, model assessment, and human review. Approve routine requests automatically and route uncertain cases to a person.',
    approvalNodes: ['Request', 'Rules / Review', 'Allow once', 'Ask a person'],
    approvalFooter: 'Rules first · Per-call authorization · Human fallback',
    continuityTitle: 'Context and state management',
    continuityItems: [
      ['Context management', 'Organize history and compact long outputs'],
      ['Project memory', 'Keep project knowledge and preferences'],
      ['Session recovery', 'Save progress and reconnect to the work'],
    ],
    demoEyebrow: '04 / BUILT-IN WEBUI',
    demoTitle: 'An integrated agent workspace',
    demoIntro:
      'Manage projects, conversations, and files in one place. Follow plans and tool calls as your agent works, and review the results as they arrive.',
    demoPlay: 'Play demo',
    demoPause: 'Pause demo',
    demoExpand: 'Enlarge screenshot',
    demoClose: 'Close image',
    demoFrames: ['Work plan', 'Execution & artifacts', 'Project files'],
    demoDetails: ['Projects & conversations', 'Live tool activity', 'Files & outputs'],
    demoCaption: 'WebUI recording · Qwen3.8-Max · Event materials to briefing and agenda',
    appsEyebrow: '05 / APPLICATIONS',
    appsTitle: 'Research | Development | Creation',
    appsIntro:
      'Specialized agents and workflows for research, software development, and video production, ready to adapt to your domain.',
    apps: [
      {
        id: 'research',
        name: 'Research & document analysis',
        label: 'Deep Research / Doc Research',
        text: 'Combine search, evidence management, and report generation for open-ended research and document analysis.',
        tags: ['Research roles', 'Evidence & reports'],
        path: 'projects/deep_research/v2',
        art: ['Question', 'Search · Evidence', 'Analysis · Synthesis', 'Research report'],
      },
      {
        id: 'code',
        name: 'Software generation',
        label: 'CodeGenesis',
        text: 'Develop projects through requirements, architecture, implementation, and verification workflows.',
        tags: ['Staged collaboration', 'Validation & refinement'],
        path: 'projects/code_genesis',
        art: ['Requirements', 'Architecture', 'Code', 'Validate'],
      },
      {
        id: 'finance',
        name: 'Financial research',
        label: 'FinResearch',
        text: 'Combine financial data, analytical methods, and reporting standards in specialized research workflows.',
        tags: ['Domain tools', 'Research standards'],
        path: 'projects/fin_research',
        art: ['Market sources', 'Collect', 'Analyze', 'Report'],
      },
      {
        id: 'cinema',
        name: 'Video creation',
        label: 'Cinema',
        text: 'Orchestrate scripts, images, voice, and video tools for content generation and post-production.',
        tags: ['Multimodal generation', 'Workflow orchestration'],
        path: 'projects/singularity_cinema',
        art: ['Script', 'Images', 'Audio', 'Film'],
      },
    ],
    viewProject: 'Explore project',
    applicationDiagram: 'Workflow diagram',
    evolutionTitle: 'Evaluation-driven skill evolution',
    evolutionText:
      'Analyze execution traces, evaluate outcomes, and revise skills through reflection. Validate candidates against tasks to retain useful improvements.',
    evolutionSteps: ['Execute', 'Evaluate', 'Reflect', 'Revise', 'Validate'],
    connectEyebrow: '07 / OPEN ECOSYSTEM',
    connectTitle: 'Open protocols, connected ecosystems',
    connectText:
      'Use your preferred models and community plugins. Connect tools and agents through standard protocols, and carry skills, instructions, and memory across frameworks.',
    getEyebrow: '08 / GET STARTED',
    getTitle: 'Build your own agent',
    getIntro:
      'Start in the WebUI, run tasks from the CLI, or integrate agents into your application with the Python SDK.',
    getTabs: ['WebUI', 'CLI', 'Python SDK'],
    copy: 'Copy code',
    copied: 'Copied',
    copyFailed: 'Could not copy. Select the code manually.',
    installHelp: [
      'Launch the workspace and configure your model.',
      'Configure a model, then run a task from the terminal.',
      'Define your agent in configuration and extend it with the SDK.',
    ],
    guide: 'Read the docs',
    sdkExample: 'Full SDK example',
    footer: 'An open-source ModelScope project',
    footerLine: 'A lightweight framework for complex tasks',
    legal: 'Apache 2.0',
    back: 'Back to top',
    evolutionEyebrow: '06 / SKILL EVOLUTION',
    evolutionCaption: 'Task feedback → Skill improvement',
    evolutionOutcome: 'Keep updates that pass validation',
    evolutionDetails: [
      ['Trace analysis', 'Identify execution issues'],
      ['Skill revision', 'Generate a candidate skill'],
      ['Validation', 'Retain effective changes'],
    ],
    modelLabel: 'Model choice',
    protocolCards: [
      {
        name: 'MCP',
        title: 'Tools and services',
        text: 'Connect external tools or expose MS-Agent tools and applications as MCP services.',
        icon: 'tool',
      },
      {
        name: 'ACP',
        title: 'Editors and agents',
        text: 'Connect ACP-compatible editors, call external agents, and route sessions through an agent proxy.',
        icon: 'terminal',
      },
      {
        name: 'A2A',
        title: 'Agent services',
        text: 'Call remote agents or publish MS-Agent as a discoverable service for other agents.',
        icon: 'team',
      },
    ],
    pluginsTitle: 'Reuse community plugins',
    pluginsText:
      'Load skills, commands, agent definitions, hooks, and MCP configuration from Claude Code, Codex, Cursor, and OpenClaw plugin formats.',
    pluginsFlow: ['Community plugins', 'Load components', 'Agent capabilities'],
    hubTitle: 'Migrate across frameworks with Agent Hub',
    hubText:
      'Convert, merge, and sync instructions, skills, and memory. Share agent resources through ModelScope Hub and reuse them across frameworks.',
    hubResources: ['Instructions', 'Skills', 'Memory'],
    ecosystemLink: 'Explore integrations',
    models: ['Qwen', 'DeepSeek', 'GLM', 'Kimi', 'OpenAI', 'Anthropic'],
    pluginFormats: ['Claude Code', 'Codex', 'Cursor', 'OpenClaw'],
    hubFrameworks: ['QwenPaw', 'OpenClaw', 'Hermes', 'Qoder', 'Nanobot', 'OpenHuman'],
  },
} as const;

export const roleCode = [
  `# general/agent.yaml
subagents: [explore, build, research]

callbacks:
  - state_inject
  - loop_guard
  - subagent_limit

subagent_limit:
  max_parallel: 4`,
  `# plan/agent.yaml
callbacks:
  - state_inject
  - plan_check
  - loop_guard

tools:
  todo_list:
    mcp: false`,
  `# research/agent.yaml
callbacks:
  - state_inject
  - round_reminder
  - stop_gate

stop_gate:
  enabled: true
  max_retries: 2`,
  `# explore/agent.yaml
tools:
  file_system:
    mcp: false
    include:
      - read_file
      - grep
      - glob`,
  `# build/agent.yaml
tools:
  file_system:
    mcp: false
    include: [read_file, write_file,
              edit_file, grep, glob]
  code_executor:
    mcp: false
    implementation: python_env`,
];
export const installCode = [
  `pip install -U "ms-agent[webui]"
ms-agent ui`,
  `pip install -U ms-agent
ms-agent run --query "Plan a research brief"`,
  `import asyncio
from ms_agent import LLMAgent
from ms_agent.config import Config

agent = LLMAgent(Config.from_task("agent.yaml"))
messages = asyncio.run(agent.run("Write a research brief"))
print(messages[-1].content)`,
];
