import type { Dict } from './i18n'

type ChatKey = keyof Dict['chat']

/** What each builtin tool DOES, keyed by its full `server---leaf` name.
 *
 * The backend already maps tool names, but to a `kind` — WHICH CARD renders the
 * call (`_tool_step_meta` in chat.py). That mapping cannot name a call for the
 * user: it says nothing at all about the tools whose calls produce no card
 * (`todo_list`, `task_control` map to None), and it deliberately merges the ones
 * whose cards are identical (grep and glob are both `search`). So this table
 * lives alongside it, doing only the naming.
 *
 * Keyed on the FULL name, never the bare leaf — the same rule the backend
 * follows, because an MCP server's `write_file` is not the builtin one.
 */
const TOOL_ACTIONS: Record<string, ChatKey> = {
  'file_system---read_file': 'toolActionFileRead',
  'file_system---write_file': 'toolActionFileWrite',
  'file_system---edit_file': 'toolActionFileEdit',
  'file_system---grep': 'toolActionGrep',
  'file_system---glob': 'toolActionGlob',
  'todo_list---todo_write': 'toolActionTodoWrite',
  'todo_list---todo_read': 'toolActionTodoRead',
  // Renders the list as markdown — a read, as far as the user is concerned.
  'todo_list---todo_render_md': 'toolActionTodoRead',
  'code_executor---shell_executor': 'toolActionShell',
  'code_executor---python_executor': 'toolActionPython',
  'code_executor---notebook_executor': 'toolActionNotebook',
  'code_executor---file_operation': 'toolActionFileOp',
  'web_search---arxiv_search': 'toolActionSearchPapers',
  'web_search---fetch_page': 'toolActionFetchPage',
  'skills---skills_list': 'toolActionSkillList',
  'skills---skill_view': 'toolActionSkillView',
  'skills---skill_manage': 'toolActionSkillManage',
  'unified_memory---memory': 'toolActionMemoryWrite',
  'unified_memory---memory_read': 'toolActionMemoryRead'
}

/** A verb phrase saying what the model is about to do with `raw`.
 *
 * Verb phrases because they slot into `toolComposingOne` ("Preparing to …" /
 * "正在准备…"), which is also why they don't reuse the step cards' labels: those
 * are Title Case headings, and "Preparing to Read file" is not a sentence.
 *
 * Anything the table doesn't know — every MCP tool, an SDK tool added after this
 * was written — reads as "call <leaf>". The leaf alone: the server prefix is the
 * noisy half (`amap-maps---maps_regeocode`), and this label lives for a moment
 * before a real card carrying the full name replaces it.
 */
export function toolActionLabel(t: Dict, raw: string): string {
  const key = TOOL_ACTIONS[raw]
  if (key) return t.chat[key]
  const sep = raw.indexOf('---')
  const server = sep > 0 ? raw.slice(0, sep) : ''
  const leaf = sep > 0 ? raw.slice(sep + 3) : raw
  // Web search names its tool after the ENGINE (`tavily_search`, `exa_search`,
  // and whatever a future install adds), so match the shape instead of listing
  // them; arxiv sits in the table above because it searches papers, not the web.
  if (
    server === 'web_search' &&
    (leaf === 'web_search' || leaf.endsWith('_search'))
  ) {
    return t.chat.toolActionSearchWeb
  }
  return t.chat.toolActionCall.replace('{name}', leaf || raw)
}
