$(document).ready(function(){
  $('#form').submit(function(event){
    event.preventDefault();

    var url = $(this).attr('action');
    var data = $("#form").serialize();

    $.getJSON(url, data, function(response){
      $("#result__score0").html((Math.round(response['score0'] * 100.0) / 100).toFixed(2));
      $("#result__score1").html((Math.round(response['score1'] * 100.0) / 100).toFixed(2));

      if(response['label'] == 0){
        $("#result__label").html('NOT ANTIBODY-REACTIVE*');
        $('#result').removeClass('result--positive').addClass('result--negative');

      } else {
        $("#result__label").html('ANTIBODY-REACTIVE**');
        $('#result').removeClass('result--negative').addClass('result--positive');
      }

      $('#result').show();
    });

    return false;
  });
});
