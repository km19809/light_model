function send() {
    /*client side */
  var chat = document.createElement("li");
  var chat_input = document.getElementById("chat_input");
  var chat_text = chat_input.value;
  chat.className = "chat-bubble mine";
  chat.innerText = chat_text
  document.getElementById("chat_list").appendChild(chat);
  chat_input.value = "";

  /* ajax request */
  var request = new XMLHttpRequest();
  request.open("POST", `${window.location.host}/api/soft`, true);
  request.onreadystatechange = function() {
    if (request.readyState !== 4 || Math.floor(request.status /100) !==2) return;
    var bot_chat = document.createElement("li");
  bot_chat.className = "chat-bubble bots";
  bot_chat.innerText = JSON.parse(request.responseText).data;
  document.getElementById("chat_list").appendChild(bot_chat);

  };
  request.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
request.send(JSON.stringify({"data":chat_text}));
}

function setDefault() {
  document.getElementById("chat_input").addEventListener("keyup", function(event) {
    let input = document.getElementById("chat_input").value;
    let button = document.getElementById("send_button");
    if(input.length>0)
    {
      button.removeAttribute("disabled");
    }
    else
    {
      button.setAttribute("disabled", "true");
    }
    // Number 13 is the "Enter" key on the keyboard
    if (event.keyCode === 13) {
      // Cancel the default action, if needed
      event.preventDefault();
      // Trigger the button element with a click
      button.click();
    }
  });
}
