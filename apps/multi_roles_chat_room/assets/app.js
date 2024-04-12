function init() {
  window.js_choose_story = function(story_id) {
    var btn = document.getElementById('entry_fake_btn');
    btn.setAttribute('data-stroy', story_id);
    if (btn) {
      btn.click();
    }
  }

  window.get_story_id = function(){
    return [document.getElementById('entry_fake_btn').getAttribute('data-stroy')]
  }
}
