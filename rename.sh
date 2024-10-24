for dir in *.csv_*; do
    newdir=$(echo "$dir" | sed 's/\.csv_/_/')
    echo "Would rename '$dir' to '$newdir'"
done


for dir in *.csv_*; do
    newdir=$(echo "$dir" | sed 's/\.csv_/_/')
    mv "$dir" "$newdir"
done
