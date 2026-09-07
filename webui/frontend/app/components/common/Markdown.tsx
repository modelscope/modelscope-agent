import { CodeHighlighter, Mermaid } from '@ant-design/x'
import { CheckOutlined } from '@ant-design/icons'
import { ConfigProvider, Typography } from 'antd'
import { useContext, useState } from 'react'
import { XMarkdown } from '@ant-design/x-markdown'
import type { ComponentProps } from '@ant-design/x-markdown'
import Latex from '@ant-design/x-markdown/plugins/Latex'
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { useTheme } from '~/lib/theme'
import { useT } from '~/lib/i18n'
import CopyIcon from '~/assets/icons/copy.svg?react'
import './Markdown.css'
// Typography themes (x-markdown-light / x-markdown-dark) are @imported in
// app.css — importing the package css here would crash Node SSR (deep css
// imports of an externalized package bypass Vite's pipeline).

interface Props {
  content: string
  /** Pass true while the content is still being streamed in. */
  streaming?: boolean
  /** Handle a leading YAML frontmatter block (```---…---```). CommonMark has
   * no frontmatter concept (and x-markdown ships no extension for it), so the
   * raw block would render as a broken heading/paragraph mix. When enabled,
   * the block is re-emitted as a fenced ```yaml code block instead. */
  frontmatter?: boolean
}

const FRONTMATTER_RE = /^---\r?\n([\s\S]*?)\r?\n---\r?\n?/

function withFrontmatterAsYaml(src: string): string {
  const m = FRONTMATTER_RE.exec(src)
  if (!m) return src
  return '```yaml\n' + m[1] + '\n```\n\n' + src.slice(m[0].length)
}

/** Flatten the ReactNode children of a mapped tag into plain text (fenced
 * code bodies arrive as text nodes / arrays of text nodes). */
function textOf(children: React.ReactNode): string {
  if (typeof children === 'string') return children
  if (Array.isArray(children)) return children.map(textOf).join('')
  return children == null ? '' : String(children)
}

/** CodeHighlighter hardcodes the prism `oneLight` palette and ignores antd's
 * darkAlgorithm — in dark mode inject `oneDark` via `highlightProps` (kept
 * transparent so the card's own background wins). */
function useHighlightProps() {
  const { theme } = useTheme()
  if (theme !== 'dark') return undefined
  return {
    style: {
      ...oneDark,
      'pre[class*="language-"]': {
        ...oneDark['pre[class*="language-"]'],
        background: 'transparent',
        margin: 0
      },
      'code[class*="language-"]': {
        ...oneDark['code[class*="language-"]'],
        background: 'transparent'
      }
    }
  }
}

/** The project's copy affordance: our glyph, our wording, neutral colour (see
 * Markdown.css). Shared by the code-block header and the Mermaid toolbar so a
 * "copy" button looks the same wherever markdown renders one.
 *
 * `Actions.Copy` (what both x components use by default) only forwards `text`
 * plus a single `icon`, leaving the COPIED state on antd's defaults — its own
 * check glyph and wording. It is an antd `Typography.Text copyable` internally,
 * so using that directly costs nothing and exposes the two-slot `icon` /
 * `tooltips` ([idle, copied]) pairs this needs. */
function CopyAction({ text }: { text: string }) {
  const { t } = useT()
  return (
    <Typography.Text
      className="msa-md-copy"
      copyable={{
        text,
        icon: [
          <CopyIcon key="idle" className="h-4 w-4" />,
          <CheckOutlined key="copied" className="text-sm !text-msa-green-5" />
        ],
        tooltips: [t.chat.copyReply, t.chat.copied]
      }}
    />
  )
}

/** Code-block header matching CodeHighlighter's built-in one (language name
 * left, copy action right) with ONE change: the copy action is ours, so its
 * glyphs and labels line up with the assistant-bubble copy button.
 *
 * A custom `header` is the only way in — CodeHighlighter exposes no icon prop
 * and hardcodes `<Actions.Copy text={code} />`.
 *
 * The header/title class names have to be reproduced for the component's own
 * stylesheet to still apply, so the prefix is resolved the same way the
 * component resolves it — antd's `ConfigContext.getPrefixCls` (which is exactly
 * what x's internal `useXProviderContext` forwards) — rather than hardcoding
 * `ant-`, which a ConfigProvider `prefixCls` would break. */
function CodeHeader({ lang, code }: { lang: string; code: string }) {
  const { getPrefixCls } = useContext(ConfigProvider.ConfigContext)
  const prefixCls = getPrefixCls('codeHighlighter')
  return (
    <div className={`${prefixCls}-header`}>
      <span className={`${prefixCls}-header-title`}>{lang}</span>
      <CopyAction text={code} />
    </div>
  )
}

/** Fenced code blocks → CodeHighlighter (language pill + copy button);
 * ```mermaid fences → live Mermaid diagrams; inline code stays plain. */
function Code(props: ComponentProps) {
  const highlightProps = useHighlightProps()
  const className = String((props as { className?: string }).className ?? '')
  const lang = /language-(\w+)/.exec(className)?.[1]
  const body = textOf(props.children)

  if (!lang) {
    // Inline code (no language- class): keep the default element.
    return <code className={className}>{props.children}</code>
  }
  if (lang === 'mermaid') {
    // Render the diagram only once the fence is complete; while streaming,
    // show the source as a code block (avoids mermaid parse churn).
    if (props.streamStatus === 'loading') {
      return (
        <CodeHighlighter
          className="msa-code-highlighter"
          lang="mermaid"
          highlightProps={highlightProps}
          header={<CodeHeader lang="mermaid" code={body} />}
        >
          {body}
        </CodeHighlighter>
      )
    }
    return <MermaidBlock code={body} />
  }
  return (
    <CodeHighlighter
      lang={lang}
      className="msa-code-highlighter"
      highlightProps={highlightProps}
      prismLightMode={false}
      header={<CodeHeader lang={lang} code={body} />}
    >
      {body}
    </CodeHighlighter>
  )
}

/** Mermaid diagram with the project's chrome.
 *
 * Two deviations from the stock component:
 *  - The built-in copy action is `Actions.Copy` (antd's default glyph/wording).
 *    `enableCopy: false` drops it and a `customActions` entry renders ours via
 *    `actionRender`, which the component supports for arbitrary JSX.
 *  - `classNames.header` hangs a hook for Markdown.css to put the toolbar band
 *    on project fill/line tokens instead of x's own palette.
 *
 * The copy action is view-dependent: stock Mermaid only offers it in CODE view
 * (image view gets zoom/download instead), and `customActions` are appended to
 * whichever set is active — so it has to be gated on the render type or it would
 * also sit next to the zoom controls, offering to copy source from a picture. */
function MermaidBlock({ code }: { code: string }) {
  const [isCode, setIsCode] = useState(false)
  return (
    <Mermaid
      classNames={{ header: 'msa-mermaid-header' }}
      onRenderTypeChange={(value) => setIsCode(String(value) === 'code')}
      actions={{
        enableCopy: false,
        customActions: isCode
          ? [{ key: 'copy', actionRender: () => <CopyAction text={code} /> }]
          : []
      }}
    >
      {code}
    </Mermaid>
  )
}

/** `<mermaid>` tag emitted by x-markdown for mermaid fences → diagram. */
function MermaidTag(props: ComponentProps) {
  return <MermaidBlock code={textOf(props.children)} />
}

/** Links always open in a NEW tab. Markdown here is model output (citations,
 * search results, docs) — navigating the SPA away from a live conversation
 * would drop the user out of the chat (and can abort an in-flight turn), so
 * every link leaves the app in a separate tab instead of in place.
 * `rel="noopener noreferrer"` because the target is untrusted content. */
function Anchor(props: ComponentProps) {
  const {
    children,
    // Dropped: x-markdown injects parser metadata that is not valid DOM.
    domNode: _domNode,
    streamStatus: _streamStatus,
    lang: _lang,
    block: _block,
    ...rest
  } = props as ComponentProps & { href?: string }
  return (
    <a {...rest} target="_blank" rel="noopener noreferrer">
      {children}
    </a>
  )
}

// Stable references (x-markdown best practice: never rebuild per render).
const COMPONENTS = {
  a: Anchor,
  code: Code,
  mermaid: MermaidTag
}
const CONFIG = { extensions: Latex() }

/**
 * Project-wide Markdown renderer. Wraps `@ant-design/x-markdown` so chat
 * messages, skill viewers, and any other consumers share a single import path
 * — making future swaps (theme tokens, plugins, custom components) one-edit
 * changes.
 *
 * Bundled capabilities (chat-oriented):
 * - GFM basics (tables, lists, links…) from x-markdown itself;
 * - fenced code → CodeHighlighter, ```mermaid → Mermaid diagrams;
 * - LaTeX math ($…$ / $$…$$) via the official Latex plugin (KaTeX);
 * - optional YAML frontmatter handling (`frontmatter` prop);
 * - light/dark typography theme following the app theme.
 */
export function Markdown({ content, streaming, frontmatter }: Props) {
  const { theme } = useTheme()
  return (
    <XMarkdown
      className={`msa-md-body ${
        theme === 'dark' ? 'x-markdown-dark' : 'x-markdown-light'
      }`}
      content={frontmatter ? withFrontmatterAsYaml(content) : content}
      components={COMPONENTS}
      config={CONFIG}
      streaming={streaming ? { hasNextChunk: true } : undefined}
    />
  )
}
