// const  SERVER_URL='http://127.0.0.1:8000';
const  SERVER_URL='http://103.206.101.253:8000';
// const  SERVER_URL='http://0.0.0.0:8000';


$('#records_table').click(function () {
    $("#records_table").show();
});


$('#faqBtn').click(function () {
    console.log("btn")
    var d = $(this).attr("data-reviewType");
    if (d == 'BERT') {
        console.log(d)
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
                    url: SERVER_URL+'/api/review',
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
    }
    else if (d == 'DISTILLBERT') {
        console.log(d)
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
                    url: SERVER_URL+'/api/ditillbertreview',
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
    }
    else {
        console.log(d)
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
                    url: SERVER_URL+'/api/mobilebertreview',
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
    }
});