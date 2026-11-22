// Landing hero background (templates/photo_hero.html): CC0 nature photographs crossfade behind the title.
// Decorative, so there are no controls or announcements. With reduced motion the first photograph stays.
// Without JavaScript the first photograph shows on its own. CSP-safe: only classes and attributes change.
(() => {
  const root = document.querySelector('[data-slideshow]');
  if (!root) return;
  const slides = [...root.querySelectorAll('.hero-slide')];
  if (slides.length < 2) return;

  const DELAY = 7000;      // time each photograph stays up
  const LOAD_WAIT = 4000;  // never hold the rotation longer than this for a slow image
  const reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  let index = Math.max(0, slides.findIndex((s) => s.classList.contains('is-active')));
  let onScreen = true;
  let timer = 0;
  let busy = false;

  // Promote data-srcset/data-src to real attributes the first time a slide is needed.
  const load = (slide) => {
    const img = slide.querySelector('img');
    if (!slide.hasAttribute('data-loaded')) {
      slide.querySelectorAll('[data-srcset]').forEach((el) => {
        el.setAttribute('srcset', el.getAttribute('data-srcset'));
        el.removeAttribute('data-srcset');
      });
      if (img.hasAttribute('data-src')) {
        img.setAttribute('src', img.getAttribute('data-src'));
        img.removeAttribute('data-src');
      }
      slide.setAttribute('data-loaded', '');
    }
    const ready = img.decode ? img.decode().catch(() => {}) : Promise.resolve();
    return Promise.race([ready, new Promise((r) => setTimeout(r, LOAD_WAIT))]);
  };

  const running = () => !reduce.matches && onScreen && !document.hidden;

  const schedule = () => {
    clearTimeout(timer);
    if (running()) timer = setTimeout(advance, DELAY);
  };

  async function advance() {
    if (busy) return;
    busy = true;
    try {
      const next = (index + 1) % slides.length;
      await load(slides[next]);
      if (!running()) return;
      slides[index].classList.remove('is-active');
      slides[next].classList.add('is-active');
      index = next;
      load(slides[(index + 1) % slides.length]);  // warm the following photograph
    } finally {
      busy = false;
      schedule();
    }
  }

  reduce.addEventListener('change', schedule);
  document.addEventListener('visibilitychange', schedule);
  if ('IntersectionObserver' in window) {
    new IntersectionObserver((entries) => {
      onScreen = entries[0].isIntersecting;
      schedule();
    }).observe(root);
  }
  if (!reduce.matches) load(slides[(index + 1) % slides.length]);
  schedule();
})();
