const nav = document.querySelector<HTMLElement>('#main-nav');
const menu = document.querySelector<HTMLButtonElement>('.menu-toggle');
function closeMenu() {
  menu?.setAttribute('aria-expanded', 'false');
  nav?.removeAttribute('data-open');
}
menu?.addEventListener('click', () => {
  const open = menu.getAttribute('aria-expanded') !== 'true';
  menu.setAttribute('aria-expanded', String(open));
  nav?.setAttribute('data-open', String(open));
});
nav?.querySelectorAll('a').forEach((a) => a.addEventListener('click', closeMenu));
document.addEventListener('keydown', (event) => {
  if (event.key === 'Escape' && menu?.getAttribute('aria-expanded') === 'true') {
    closeMenu();
    menu.focus();
  }
});

document.querySelectorAll<HTMLElement>('[data-tab-group]').forEach((group) => {
  const tabs = Array.from(group.querySelectorAll<HTMLButtonElement>('[role=tab]'));
  const activate = (tab: HTMLButtonElement, focus = false) => {
    tabs.forEach((t) => {
      const selected = t === tab;
      t.setAttribute('aria-selected', String(selected));
      t.tabIndex = selected ? 0 : -1;
    });
    group.querySelectorAll<HTMLElement>('[data-panel]').forEach((panel) => {
      panel.hidden = panel.dataset.panel !== tab.dataset.tab;
    });
    if (focus) tab.focus();
  };
  tabs.forEach((tab, i) => {
    tab.addEventListener('click', () => activate(tab));
    tab.addEventListener('keydown', (event) => {
      let index: number | undefined;
      if (event.key === 'ArrowRight' || event.key === 'ArrowDown') index = (i + 1) % tabs.length;
      if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') index = (i - 1 + tabs.length) % tabs.length;
      if (event.key === 'Home') index = 0;
      if (event.key === 'End') index = tabs.length - 1;
      if (index !== undefined) {
        event.preventDefault();
        activate(tabs[index], true);
      }
    });
  });
});

document.querySelectorAll<HTMLButtonElement>('[data-copy]').forEach((button) => {
  let timer: ReturnType<typeof setTimeout>;
  button.addEventListener('click', async () => {
    const code = button.parentElement?.querySelector('code')?.textContent;
    const label = button.querySelector('span');
    if (!code || !label) return;
    let message = '';
    try {
      await navigator.clipboard.writeText(code);
      message = button.dataset.done || 'Copied';
    } catch {
      message = button.dataset.error || 'Could not copy';
    }
    label.textContent = message;
    const live = document.querySelector('#announcer');
    if (live) live.textContent = message;
    clearTimeout(timer);
    timer = setTimeout(() => {
      label.textContent = button.dataset.label || 'Copy';
    }, 2200);
  });
});

const video = document.querySelector<HTMLVideoElement>('#demo-video');
const picture = document.querySelector<HTMLButtonElement>('[data-enlarge]');
const screenshot = document.querySelector<HTMLImageElement>('#demo-image');
const play = document.querySelector<HTMLButtonElement>('[data-play]');
const dialog = document.querySelector<HTMLDialogElement>('.image-dialog');
function playLabel(playing: boolean) {
  const label = play?.querySelector('span');
  if (label && play) label.textContent = (playing ? play.dataset.pauseLabel : play.dataset.playLabel) || '';
}
document.querySelectorAll<HTMLButtonElement>('[data-frame]').forEach((button) =>
  button.addEventListener('click', () => {
    if (!screenshot || !picture || !video) return;
    video.pause();
    video.hidden = true;
    picture.hidden = false;
    screenshot.src = button.dataset.frame || '';
    document
      .querySelectorAll<HTMLButtonElement>('[data-frame]')
      .forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    playLabel(false);
  }),
);
play?.addEventListener('click', async () => {
  if (!video || !picture) return;
  if (!video.hidden && !video.paused) {
    video.pause();
    return;
  }
  const source = video.querySelector('source');
  if (source && !source.hasAttribute('src')) {
    source.src = source.dataset.src || '';
    video.load();
  }
  video.hidden = false;
  picture.hidden = true;
  try {
    await video.play();
  } catch {
    video.controls = true;
    playLabel(false);
  }
});
video?.addEventListener('play', () => playLabel(true));
video?.addEventListener('pause', () => playLabel(false));
video?.addEventListener('ended', () => playLabel(false));
picture?.addEventListener('click', () => {
  const img = dialog?.querySelector('img');
  if (img && screenshot) img.src = screenshot.src;
  dialog?.showModal();
});
dialog?.querySelector('button')?.addEventListener('click', () => dialog.close());
dialog?.addEventListener('click', (event) => {
  if (event.target === dialog) dialog.close();
});
