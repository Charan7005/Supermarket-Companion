$(document).ready(function() {
    $('#search-btn').click(function() {
        searchProduct();
    });
});

function searchProduct() {
    var query = $('#search-input').val().trim();
    if (query !== '') {
        $.ajax({
            type: 'GET',
            url: '/search',
            data: { q: query },
            success: function(response) {
                displaySearchResults(response);
            },
            error: function(xhr, status, error) {
                console.error('Error searching for product:', error);
            }
        });
    }
}

function displaySearchResults(results) {
    var searchResultsContainer = $('#search-results');
    searchResultsContainer.empty();

    if (results.length > 0) {
        var resultList = '<ul>';
        results.forEach(function(result) {
            resultList += '<li>' + result.name + ' - $' + result.price + '</li>';
        });
        resultList += '</ul>';
        searchResultsContainer.html(resultList);
    } else {
        searchResultsContainer.html('<p>No results found.</p>');
    }
}
