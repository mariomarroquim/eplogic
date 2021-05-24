$(document).ready(function(){
  $('#form').submit(function(event){
    event.preventDefault();

    var url = $(this).attr('action');
    var data = $("#form").serialize();

    $.getJSON(url, data, function(response){
      $("#result__score").html((Math.round(response['score'] * 100.0) / 100).toFixed(2));

      if(response['label'] == 0){
        $("#result__label").html('NEGATIVE');
        $('#result').removeClass('result--positive').addClass('result--negative');

      } else {
        $("#result__label").html('POSITIVE');
        $('#result').removeClass('result--negative').addClass('result--positive');
      }

      $('#result').show();
    });

    return false;
  });
});
