// @Sergio_Gordillo.
'use strict';
(function() {

	const IMG_FOLDER = 'static/data';
	const NUM_IMG = 20;
	const NUM_SAMPLES = 200

	const imageContainer = document.getElementById('image-container');
	const offset = Math.round(Math.random() * (NUM_SAMPLES-NUM_IMG));

	let imageId;

	function getImageSample(index){
		let snippet = "<img id=\"image-sample-"+index+"\" src=\"static/data/"+index+".png\">";
		return snippet;
	}

	function predict(){
		if(typeof imageId === "undefined"){
			alert('Please, select an image!')
		}
		else{
			try{
				var xmlhttp = new XMLHttpRequest();

				xmlhttp.onreadystatechange = function() {
		        if (xmlhttp.readyState == XMLHttpRequest.DONE ) {
		           if (xmlhttp.status == 200) {
		               console.log(xmlhttp.responseText);
		           }
		           else if (xmlhttp.status == 400) {
		              alert('There was an error 400');
		           }
		           else {
		               alert('something else other than 200 was returned');
		           }
		        }
		    };
				xmlhttp.open("POST", "/api/predict", true);
				xmlhttp.setRequestHeader("Content-Type", "application/json;charset=UTF-8");
				xmlhttp.send(JSON.stringify({"imageId":imageId}));
			}
			catch(e){
				// Manage Error
			}
		}
	}


	(function init(){
		// Show random images
		for(var i=0; i < NUM_IMG; i++){ // images name starts at 1
			imageContainer.innerHTML += getImageSample(offset+i);
		}
		// Assign behavior
		var images = imageContainer.getElementsByTagName('img');
		[].forEach.call(images, function(image){
			image.addEventListener('click', function(e){
				var id = this.id;
				imageId = id.split("-")[id.split("-").length-1];
			});
		});
		// Assign behavior
		var button = document.getElementById('predict-button');
		button.addEventListener('click', function(e){
			predict();
		});
	})();

})(); // End of use strict
