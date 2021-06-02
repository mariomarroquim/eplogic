$(document).ready(function(){
  $("#set-example").on( "click", function(event) {
    event.preventDefault();

    $("#eplet_locus").val("abc").change();
    $("#eplet_allele_qtd").val("17");
    $("#eplet_min_mfi").val("509");
    $("#eplet_max_mfi").val("12000");
    $("#panel_nc").val("102");
    $("#panel_pc").val("13200");

    $("#continue").fadeOut(100).fadeIn(200).fadeOut(100).fadeIn(200);
  });

  $("#form").submit(function(event){
    event.preventDefault();

    $("#continue").html("Please, wait...");

    var url = $(this).attr("action");
    var data = $("#form").serialize();

    $.getJSON(url, data, function(response){
      $("#result__score0").html((Math.round(response["score0"] * 100.0)).toFixed(2) + "%.");
      $("#result__score1").html((Math.round(response["score1"] * 100.0)).toFixed(2) + "%.");

      if(response["label"] == 0){
        $("#result__label").html("IS NOT");
        $("#result").removeClass("result--positive").addClass("result--negative");

      } else {
        $("#result__label").html("IS");
        $("#result").removeClass("result--negative").addClass("result--positive");
      }

      $("#result").show();
    }).fail(function() { alert("EpLogic could not continue.\n\nPlease, check if all fields are filled."); });

    $("#continue").html("Continue");

    return false;
  });
});
