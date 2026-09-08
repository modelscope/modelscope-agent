import { createReadableStreamFromReadable } from '@react-router/node'
import { isbot } from 'isbot'
import { PassThrough } from 'node:stream'
import type { RenderToPipeableStreamOptions } from 'react-dom/server'
import { renderToPipeableStream } from 'react-dom/server'
import type { EntryContext } from 'react-router'
import { ServerRouter } from 'react-router'

const STREAM_TIMEOUT = 5_000

export default function handleRequest(
  request: Request,
  responseStatusCode: number,
  responseHeaders: Headers,
  routerContext: EntryContext
) {
  return new Promise<Response>((resolve, reject) => {
    let shellRendered = false
    const userAgent = request.headers.get('user-agent')

    // Bots and SPA mode renders wait for all content; humans get a fast shell.
    // See https://react.dev/reference/react-dom/server/renderToPipeableStream
    const readyOption: keyof RenderToPipeableStreamOptions =
      (userAgent && isbot(userAgent)) || routerContext.isSpaMode
        ? 'onAllReady'
        : 'onShellReady'

    const { pipe, abort } = renderToPipeableStream(
      <ServerRouter context={routerContext} url={request.url} />,
      {
        [readyOption]() {
          shellRendered = true

          // Nothing to post-process: antd/x styles are pre-baked into the
          // stylesheet the root loader links (see scripts/genAntdCss.tsx), so
          // the HTML can stream straight through instead of being buffered to
          // have extracted CSS spliced into <head>.
          const body = new PassThrough()

          responseHeaders.set('Content-Type', 'text/html')

          resolve(
            new Response(createReadableStreamFromReadable(body), {
              headers: responseHeaders,
              status: responseStatusCode
            })
          )

          pipe(body)
        },
        onShellError(error: unknown) {
          reject(error)
        },
        onError(error: unknown) {
          responseStatusCode = 500
          // Shell errors get rejected above and logged by the framework; only
          // log post-shell streaming errors here.
          if (shellRendered) {
            console.error(error)
          }
        }
      }
    )

    setTimeout(abort, STREAM_TIMEOUT + 1000)
  })
}
