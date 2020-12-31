$('#records_table').click(function () {
    $("#records_table").show();
});


$('#faqBtn').click(function () {
    console.log("btn")
    var textboxvalue = $('#review').val();
    console.log(textboxvalue)
    console.log(textboxvalue.length)

    if ($.trim(textboxvalue).length > 0) {

        $('#records_table > tbody').empty();

        $("#circle").show();
        $("#faqBtn").hide();

        console.log(textboxvalue)
        $.ajax(
            {
                type: "POST",
                url: 'http://127.0.0.1:8000/review',
                data: { doc: textboxvalue },
                success: function (result) {
                    $("#records_table").show();
                    $("#circle").hide();
                    $("#faqBtn").show();

                    console.log(result)
                    var trHTML = '';
                    $.each(result, function (i, item) {
                        trHTML += '<tr><td>' + i + '</td><td>' + item + '</td></tr>';

                    });
                    $('#records_table').append(trHTML);
                }
            });
    }
    else {
        $('#records_table > tbody').empty();

    }
});