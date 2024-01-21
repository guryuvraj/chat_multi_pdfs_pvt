

css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
    max-width: calc(100% - 67px);
  overflow-wrap: break-word;
}
.chat-message.user {
    background-color: #2b313e

}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.ibb.co/h19qbYk/bot-read.gif">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://avatars.githubusercontent.com/u/98197801?s=400&u=14857448bbced45b7ef9e3209188b26c44596d3a&v=4">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''