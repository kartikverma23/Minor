$(document).ready(function () {
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            };
            reader.readAsDataURL(input.files[0]);
        }
    }

    $('#imageUpload').change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
        $('#result').text('').hide();
        readURL(this);
    });

    $('.upload-label').click(function () {
        $('.webcam').hide();
    });

    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);
        $(this).hide();
        $('.loader').show();

        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (predictions) {
                $('#result').fadeIn(600);
                $('.loader').hide();
                if (!predictions || predictions.length === 0) {
                    $('#result').text('Result: No landmarks detected.');
                } else {
                    $('.image-section').hide();
                    $('#result').html('<img src="data:image/jpeg;base64,' + predictions + '"/>');
                }
                console.log('Success!');
            },
            error: function (xhr) {
                $('.loader').hide();
                $('#btn-predict').show();
                $('#result').text(xhr.responseText || 'Prediction failed.');
                $('#result').show();
            }
        });
    });
});
