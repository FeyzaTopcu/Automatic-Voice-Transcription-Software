try {
    var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    var recognition = new SpeechRecognition();
} catch (e) {
    console.error(e);
    $('.no-browser-support').show();
    $('.app').hide();
}


var noteText = $('#note-textarea');
var instructions = $('#recording-instructions');
var List = $('ul#notes');

var content = '';


var notes = getAllNotes();
renderNotes(notes);



recognition.continuous = true;


recognition.onresult = function(event) {

    var current = event.resultIndex;


    var transcript = event.results[current][0].transcript;


    var mobileRepeatBug = (current == 1 && transcript == event.results[0][0].transcript);

    if (!mobileRepeatBug) {
        content += transcript;
        noteText.val(content);
    }
};

recognition.onstart = function() {
    instructions.text('Voice recognition activated. Try speaking into the microphone.');
}

recognition.onspeechend = function() {
    instructions.text('You were quiet for a while so voice recognition turned itself off.');
}

recognition.onerror = function(event) {
    if (event.error == 'no-speech') {
        instructions.text('No speech was detected. Try again.');
    };
}




$('#start-record-btn').on('click', function(e) {
    if (content.length) {
        content += ' ';
    }
    recognition.start();
});


$('#pause-record-btn').on('click', function(e) {
    recognition.stop();
    instructions.text('Voice recognition paused.');
});


noteText.on('input', function() {
    content = $(this).val();
})

$('#save-note-btn').on('click', function(e) {
    recognition.stop();

    if (!content.length) {
        instructions.text('Could not save empty note. Please add a message to your note.');
    } else {

        saveNote(new Date().toLocaleString(), content);


        content = '';
        renderNotes(getAllNotes());
        noteText.val('');
        instructions.text('Note saved successfully.');
    }

})


List.on('click', function(e) {
    e.preventDefault();
    var target = $(e.target);


    if (target.hasClass('listen-note')) {
        var content = target.closest('.note').find('.content').text();
        readOutLoud(content);
    }


    if (target.hasClass('delete-note')) {
        var dateTime = target.siblings('.date').text();
        deleteNote(dateTime);
        target.closest('.note').remove();
    }
});




function readOutLoud(message) {
    var speech = new SpeechSynthesisUtterance();


    speech.text = message;
    speech.volume = 1;
    speech.rate = 1;
    speech.pitch = 1;

    window.speechSynthesis.speak(speech);
}




function renderNotes(notes) {
    var html = '';
    if (notes.length) {
        notes.forEach(function(note) {
            html += `<li class="note">
        <p class="header">
          <span class="date">${note.date}</span>
          <a href="#" class="listen-note" title="Listen to Note">Listen to Note</a>
          <a href="#" class="delete-note" title="Delete">Delete</a>
        </p>
        <p class="content">${note.content}</p>
      </li>`;
        });
    } else {
        html = '<li><p class="content">You don\'t have any notes yet.</p></li>';
    }
    List.html(html);
}


function saveNote(dateTime, content) {
    localStorage.setItem('note-' + dateTime, content);
}


function getAllNotes() {
    var notes = [];
    var key;
    for (var i = 0; i < localStorage.length; i++) {
        key = localStorage.key(i);

        if (key.substring(0, 5) == 'note-') {
            notes.push({
                date: key.replace('note-', ''),
                content: localStorage.getItem(localStorage.key(i))
            });
        }
    }

    var fs = require('fs');
    fs.writeFile('gyDosya.txt', notes.text, function(err) {
        if (err) return console.log(err);
        console.log('gyDosya.txt NIN ICERIGI DEGISTI');
    });

    fs.readFile("gyDosya.txt", "utf8", function(hata, notes) {

        console.log(notes);
    });

    return notes;
}


function deleteNote(dateTime) {
    localStorage.removeItem('note-' + dateTime);
}